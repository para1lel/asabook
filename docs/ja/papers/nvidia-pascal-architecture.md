---
title: 'NVIDIA Tesla P100 Pascal Architecture'
createTime: 2026/09/13 13:21:06
permalink: /ja/papers/nvidia-pascal-architecture/
pageClass: paper-reading
---

> [NVIDIA Corporation](https://www.nvidia.com/en-us/about-nvidia/)。*[NVIDIA Tesla P100: The Most Advanced Datacenter Accelerator Ever Built](https://www.nvidia.com/en-us/data-center/resources/pascal-architecture-whitepaper/)*。当時世界最速の GPU であった Pascal GP100 を搭載。V1.2、2017 年。本ホワイトペーパーは Tesla P100 の発表とともに 2016 年 4 月 5 日に初公開された。<a href="/paper/nvidia-pascal-architecture.pdf" target="_blank" rel="noopener noreferrer">原 PDF</a>。arXiv レコードおよび TeX ソースは存在しないため、正確な文言、印刷レイアウト、参考文献については公開 PDF を正本とする。

<span id="section-1"></span>

## 1 はじめに

10 年近く前、NVIDIA® は G80 GPU と NVIDIA® CUDA® 並列コンピューティングプラットフォームの導入により、GPU を使って計算負荷の高いワークロードを高速化する手法を切り拓いた。現在、NVIDIA® Tesla® GPU は、数値流体力学、医学研究、マシンビジョン、金融モデリング、量子化学、エネルギー探査など、多数の分野にわたる数千の High Performance Computing（HPC）アプリケーションを高速化している。

NVIDIA Tesla GPU は世界有数のスーパーコンピューターの多くに搭載され、発見を速めるとともに、複数分野でますます複雑になるシミュレーションを可能にしている。データセンターは NVIDIA Tesla GPU を使って多数の HPC および Big Data アプリケーションを高速化し、同時に最先端の Artificial Intelligence（AI）と Deep Learning システムを実現している。

画期的な NVIDIA® Pascal™ GP100 GPU を採用した NVIDIA の新しい NVIDIA Tesla P100 アクセラレーター（[図 1](#figure-01)）は、GPU コンピューティングを次の段階へ進める。本稿では Tesla P100 アクセラレーターと Pascal GP100 GPU の両アーキテクチャを詳述する。

8 基の Tesla P100 アクセラレーターを使用する NVIDIA の強力な新型 DGX-1 サーバー、すなわち事実上「1 台の箱に収めた AI スーパーコンピューター」についても論じる。DGX-1 は、AI を前進させる研究者と、Deep Learning のための統合システムを必要とするデータサイエンティストを支援する目的で設計された。

<span id="figure-01"></span>

![Pascal GP100 GPU を搭載した NVIDIA Tesla P100 アクセラレーター](../../papers/nvidia-pascal-architecture/figure-01.png)

**図 1。** Pascal GP100 GPU を搭載した NVIDIA Tesla P100

<span id="section-2"></span>

## 2 Tesla P100：GPU コンピューティングのための革新的な性能と機能

153 億トランジスタの GPU、GPU 間および GPU-CPU 間通信を大幅に高速化する新しい高性能インターコネクト、GPU プログラミングを簡素化する新技術、そして卓越した電力効率を備える Tesla P100 は、史上最も強力なだけでなく、アーキテクチャ上最も複雑な GPU アクセラレーターでもある。

Tesla P100 の主な機能は次のとおりである。

- **極限の性能**<br>
  HPC、Deep Learning、その他多くの GPU コンピューティング分野を駆動
- **NVLink™**<br>
  アプリケーションの最大限のスケーラビリティを実現する NVIDIA の新しい高速・広帯域インターコネクト
- **HBM2**<br>
  高速、大容量、きわめて高効率な CoWoS（Chip-on-Wafer-on-Substrate）積層メモリアーキテクチャ
- **Unified Memory、Compute Preemption、新しい AI アルゴリズム**<br>
  大幅に改善されたプログラミングモデルと、Pascal アーキテクチャ向けに最適化された高度な AI ソフトウェア
- **16nm FinFET**<br>
  より多くの機能、より高い性能、改善された電力効率を実現

<span id="figure-02"></span>

![Tesla P100 の 5 つの新技術](../../papers/nvidia-pascal-architecture/figure-02.png)

**図 2。** Tesla P100 の新技術

<span id="section-2-1"></span>

### 2.1 High Performance Computing と Deep Learning のための極限の性能

Tesla P100 は、最も要求の厳しいコンピュートアプリケーションに卓越した性能を提供する目的で構築され、次の性能を実現する。

- 5.3 TFLOPS の倍精度浮動小数点（FP64）性能
- 10.6 TFLOPS の単精度（FP32）性能
- 21.2 TFLOPS の半精度（FP16）性能

<span id="figure-03"></span>

![Tesla P100 と過去世代 GPU の計算性能比較](../../papers/nvidia-pascal-architecture/figure-03.png)

**図 3。** Tesla P100 の計算性能は過去世代の GPU を大幅に上回る

NVIDIA GPU は長年にわたり High Performance Computing の多数の分野を高速化してきたが、最近では特に Deep Learning が GPU 高速化の非常に重要な焦点となっている。NVIDIA GPU は現在、Deep Neural Network（DNN）と Artificial Intelligence（AI）の最前線にある。さまざまなアプリケーションで DNN を CPU の 10～20 倍に高速化し、訓練時間を数週間から数日へ短縮している。過去 3 年間で、NVIDIA GPU ベースのコンピューティングプラットフォームは Deep Learning ネットワークの訓練時間を 50 倍高速化した。過去 2 年間で、NVIDIA と Deep Learning に取り組む企業数は約 35 倍に増え、3,400 社を超えた。

ネイティブ 16-bit 浮動小数点（FP）精度など、Pascal アーキテクチャの新しい技術により、GP100 は多くの Deep Learning アルゴリズムを大幅に高速化できる。これらのアルゴリズムは高い浮動小数点精度を必要としない一方、FP16 がもたらす追加の計算能力と、16-bit データ型による記憶容量の削減から大きな恩恵を受ける。

<span id="section-2-2"></span>

### 2.2 NVLink：マルチ GPU および GPU-CPU 接続のための卓越した帯域幅

GPU アクセラレーテッドコンピューティングの普及に伴い、ワークステーション、サーバー、スーパーコンピューターのあらゆる階層で、より多くのマルチ GPU システムが導入されている。多くの 4-GPU および 8-GPU システム構成が、より大規模で複雑な問題を解くために使われている。複数のマルチ GPU システム群を InfiniBand® と 100 Gb Ethernet で相互接続し、はるかに大規模で強力なシステムを構成している。GPU と CPU の比率も高まった。Oak Ridge National Labs に設置された 2012 年最速のスーパーコンピューター Titan は、CPU ごとに 1 基の GK110 GPU を配置していた。現在では、開発者がアプリケーション中の並列性をさらに引き出して活用するにつれ、CPU ごとに 2 基以上の GPU を組み合わせることが一般的になっている。この傾向が続くと、マルチ GPU システムの PCIe 帯域幅はさらに大きなボトルネックになる。

この問題に対処するため、Tesla P100 は NVIDIA の新しい高速インターフェース NVLink を備え、GPU 間データ転送に最大 160 Gigabytes/second の双方向帯域幅、すなわち PCIe Gen 3 x16 の 5 倍を提供する。[図 4](#figure-04) は、NVLink が 8 基の Tesla P100 アクセラレーターを Hybrid Cube Mesh Topology で接続する様子を示す。

<span id="figure-04"></span>

![Hybrid Cube Mesh で接続された 8 基の Tesla P100 アクセラレーター](../../papers/nvidia-pascal-architecture/figure-04.png)

**図 4。** NVLink が 8 基の Tesla P100 アクセラレーターを Hybrid Cube Mesh Topology で接続

[図 5](#figure-05) は各種ワークロードの性能を示し、NVLink で接続した最大 8 基の GP100 GPU によってサーバーが達成できる性能スケーラビリティを明らかにする。（注：これらの数値は量産前の P100 GPU で測定した。）

<span id="figure-05"></span>

![NVLink で接続した 1、2、4、8 基の P100 GPU によるワークロード高速化](../../papers/nvidia-pascal-architecture/figure-05.png)

**図 5。** NVLink で接続した 8 基の P100 による最大の性能向上

<span id="section-2-3"></span>

### 2.3 HBM2 高速 GPU メモリアーキテクチャ

Tesla P100 は HBM2 メモリに対応した世界初の GPU アーキテクチャである。HBM2 は Maxwell GM200 GPU の 3 倍のメモリ帯域幅を提供する。これにより P100 は、はるかに大きなデータのワーキングセットをより高い帯域幅で扱い、効率と計算スループットを高め、システムメモリからの転送頻度を減らせる。

HBM2 は積層メモリであり、GPU と同じ物理パッケージに配置されるため、従来の GDDR5 と比べて大幅に省スペース化でき、従来以上に高密度な GPU サーバーを容易に構築できる。

<span id="figure-06"></span>

![Tesla P100 HBM2 と過去世代 GPU のメモリ帯域幅比較](../../papers/nvidia-pascal-architecture/figure-06.png)

**図 6。** HBM2 を搭載する Tesla P100 は過去世代 GPU のメモリ帯域幅を大幅に上回る

<span id="section-2-4"></span>

### 2.4 Unified Memory と Compute Preemption による開発者向けプログラミングの簡素化

Unified Memory は NVIDIA GPU コンピューティングにおける大きな進歩であり、Pascal GP100 GPU アーキテクチャの主要な新ハードウェア・ソフトウェア機能でもある。CPU と GPU のメモリに対して単一でシームレスな統合仮想アドレス空間を提供する。Unified Memory は GPU プログラミングとアプリケーションの GPU 移植を大幅に簡素化し、GPU コンピューティングの学習曲線も緩やかにする。プログラマーは、異なる 2 つの仮想メモリシステム間のデータ共有管理を気にする必要がなくなる。

GP100 はハードウェアページフォールトに対応した最初の NVIDIA GPU であり、新しい 49-bit（512 TB）仮想アドレス指定と組み合わせることで、GPU と CPU の完全な仮想アドレス空間間で透過的にデータを移行できる。

Compute Preemption は GP100 に追加されたもう 1 つの重要なハードウェア・ソフトウェア機能であり、従来の Maxwell および Kepler GPU アーキテクチャのスレッドブロック単位ではなく、命令単位で計算タスクをプリエンプトできる。Compute Preemption は、長時間実行するアプリケーションによるシステムの独占（他のアプリケーションの実行妨害）やタイムアウトを防ぐ。プログラマーは、長時間実行するアプリケーションを他の GPU アプリケーションと共存させるために変更する必要がなくなる。GP100 の Compute Preemption により、アプリケーションは大規模データセットの処理や各種条件の成立待ちに必要なだけ実行しながら、他のタスクと並行してスケジュールできる。たとえば、インタラクティブなグラフィックス処理とデバッガーを、長時間実行する計算タスクと同時に動かせる。

<span id="section-3"></span>

## 3 GP100 GPU ハードウェアアーキテクチャの詳細

GP100 は、Tesla P100 アクセラレータープラットフォームが対象とする GPU アクセラレーテッドコンピューティング市場の要求に応える、世界最高性能の並列コンピューティングプロセッサーとして構築された。従来の Tesla クラス GPU と同様、GP100 は Graphics Processing Cluster（GPC）、Texture Processing Cluster（TPC）、Streaming Multiprocessor（SM）、メモリコントローラーの配列から構成される。完全な GP100 は、6 基の GPC、60 基の Pascal SM、30 基の TPC（各 TPC は 2 基の SM を含む）、8 基の 512-bit メモリコントローラー（合計 4096 bit）を備える。

GP100 内の各 GPC は 10 基の SM を持つ。各 SM は 64 基の CUDA Core と 4 基のテクスチャユニットを持つ。60 基の SM により、GP100 は合計 3840 基の単精度 CUDA Core と 240 基のテクスチャユニットを備える。各メモリコントローラーは 512 KB の L2 キャッシュに接続され、各 HBM2 DRAM スタックは 2 基のメモリコントローラーによって制御される。完全な GPU は合計 4096 KB の L2 キャッシュを備える。

[図 7](#figure-07) は 60 基の SM ユニットを持つ完全な GP100 GPU を示す（製品によって GP100 の構成は異なる）。Tesla P100 アクセラレーターは 56 基の SM ユニットを使用する。

<span id="figure-07"></span>

![60 基の SM ユニットを持つ完全な Pascal GP100 GPU のブロック図](../../papers/nvidia-pascal-architecture/figure-07.png)

**図 7。** 60 基の SM ユニットを持つ完全な Pascal GP100 GPU

<span id="section-3-1"></span>

### 3.1 卓越した性能と電力効率

性能向上と電力効率の改善は、新しい GPU アーキテクチャにおける 2 つの主要目標である。Maxwell アーキテクチャでは SM に複数の変更を施し、Kepler より効率を高めた。Pascal はこれを土台として、Maxwell よりワット当たり性能をさらに高める改良を組み込んでいる。TSMC の 16-nm FinFET 製造プロセスが重要な役割を果たす一方、高性能を維持しながら消費電力をさらに抑えるため、多数の GPU アーキテクチャ変更も実施した。

<span id="table-01"></span>

![Tesla K40、Tesla M40、Tesla P100 の仕様](../../papers/nvidia-pascal-architecture/table-01.png)

**表 1。** Tesla P100 と前世代 Tesla 製品の比較

<span id="section-3-2"></span>

### 3.2 Pascal Streaming Multiprocessor

GP100 の第 6 世代 SM アーキテクチャは CUDA Core の利用率と電力効率を高め、GPU 全体の性能を大幅に向上させるとともに、従来の GPU より高いコアクロックを可能にする。

GP100 の SM は 64 基の単精度（FP32）CUDA Core を組み込む。対して Maxwell と Kepler の SM は、それぞれ 128 基と 192 基の FP32 CUDA Core を備えていた。GP100 SM は 2 つの処理ブロックに分かれ、各ブロックは 32 基の単精度 CUDA Core、命令バッファ、warp scheduler、2 基の dispatch unit を備える。GP100 SM の CUDA Core 総数は Maxwell SM の半分だが、同じレジスターファイル容量を維持し、同程度の warp と thread block occupancy を支える。GP100 SM のレジスター数は Maxwell GM200 および Kepler GK110 SM と同じだが、GP100 GPU 全体では SM がはるかに多いため、総レジスター数も大幅に多い。これは GPU 全体のスレッドがより多くのレジスターを利用でき、GP100 が従来世代の GPU より多くのスレッド、warp、thread block を同時に処理できることを意味する。

SM 数が増えたことで GP100 GPU 全体の shared memory も増加し、aggregate shared memory bandwidth は実質的に 2 倍以上になった。GP100 では SM 当たりの shared memory、レジスター、warp の比率が高まり、SM はコードをより効率よく実行できる。命令スケジューラーが選択できる warp が増え、より多くの load を開始でき、スレッド当たりの shared memory bandwidth も高くなる。

[図 8](#figure-08) に GP100 SM のブロック図を示す。

Kepler と比べ、Pascal の SM はより単純な datapath 構成を採用し、SM 内のデータ転送管理に必要なダイ面積と電力を削減する。Pascal はさらに優れたスケジューリングと重複実行する load/store 命令により、浮動小数点ユニットの利用率を高める。GP100 の新しい SM scheduler アーキテクチャは Maxwell scheduler の進歩をさらに発展させ、より高度な判断によって性能を高めながら消費電力を抑える。各 warp scheduler（処理ブロックごとに 1 基）は、1 クロック当たり 2 つの warp 命令を dispatch できる。

GP100 の FP32 CUDA Core に追加された新機能の 1 つは、本稿で後述するように、16-bit と 32-bit の両方の精度の命令とデータを処理できることである。FP16 演算スループットは FP32 の最大 2 倍である。

<span id="figure-08"></span>

![Pascal GP100 Streaming Multiprocessor のブロック図](../../papers/nvidia-pascal-architecture/figure-08.png)

**図 8。** Pascal GP100 SM ユニット

<span id="section-3-3"></span>

### 3.3 高性能倍精度向け設計

倍精度演算は、線形代数、数値シミュレーション、量子化学など多くの HPC アプリケーションの中核をなす。したがって、これらのユースケースにおける実効性能を大幅に高めることが、GP100 の主要な設計目標の 1 つだった。

GP100 の各 SM は 32 基の倍精度（FP64）CUDA Core を備え、これは FP32 単精度 CUDA Core の半数である。完全な GP100 GPU は 1920 基の FP64 CUDA Core を持つ。単精度（SP）ユニットと倍精度（DP）ユニットの 2：1 という比率は GP100 の新しい datapath 構成により適合し、GPU が DP ワークロードをより効率よく処理できる。従来の GPU アーキテクチャと同様、GP100 は fused multiply-add（FMA）演算と denormalized value の全速処理を含め、IEEE 754-2008 準拠の単精度および倍精度演算を完全に支援する。

> **注：** Kepler GK110 における SP ユニットと DP ユニットの比率は 3：1 だった。

<span id="section-3-4"></span>

### 3.4 FP16 演算対応による Deep Learning の高速化

Deep Learning は最も急速に成長しているコンピューティング分野の 1 つである。リアルタイム言語翻訳、高精度画像認識、自動画像キャプション、自動運転での物体認識、最適経路計算、衝突回避など、多くの重要なアプリケーションに不可欠である。Deep Learning は 2 段階のプロセスである。

- 最初に、ニューラルネットワークを訓練する。
- 次に、ネットワークを現場に配備して推論計算を実行し、過去の訓練結果を使って未知の入力を分類、認識、処理する。

GPU は CPU と比べ、Deep Learning の訓練と推論を大幅に高速化できる。

高精度浮動小数点計算を必要とする他のテクニカルコンピューティングアプリケーションと異なり、Deep Neural Network アーキテクチャは、訓練で使う backpropagation アルゴリズムにより誤差に対する自然な耐性を持つ。実際、ネットワークが訓練データセットに過剰適合することを避けるため、dropout などの手法は、訓練済みネットワークが十分に汎化し、特定ユニットの計算精度（または誤差）へ過度に依存しないようにすることを目指す。

FP32 や FP64 より FP16 でデータを格納すると、ニューラルネットワークのメモリ使用量が減り、より大きなネットワークを訓練・配備できる。FP16 計算は FP32 演算に比べて性能を最大 2 倍に高め、FP16 データ転送も FP32 や FP64 より短時間で済む。

> **注：** GP100 では、1 つの paired-operation 命令で 2 つの FP16 演算を実行できる。

GP100 のアーキテクチャ改良と FP16 データ型への対応を組み合わせることで、わずか 1 年前に可能だった水準と比べ、Deep Learning の処理時間を大幅に短縮できる。

<span id="section-3-5"></span>

### 3.5 Atomics の改善

atomic memory operation は並列プログラミングで重要であり、並行スレッドが共有データ構造に対して read-modify-write 操作を正しく実行できるようにする。

Kepler の shared memory atomic operation は Fermi と同じ形式だった。両アーキテクチャは lock/update/unlock パターンを使って shared memory atomics を実装しており、shared memory の特定位置に更新が集中すると高コストになる場合があった。

Maxwell は 32-bit 整数用 shared memory atomic operation と、他の atomic function を Fermi や Kepler のソフトウェア方式より低いオーバーヘッドで実装できる 32-bit および 64-bit compare-and-swap（CAS）を shared memory のネイティブハードウェアで支援し、atomic operation を改善した。

GP100 は Maxwell を土台として、新しい Unified Memory と NVLink の機能（以降の段落で説明）によって atomic operation をさらに改善する。global memory の atomic addition は FP64 データにも拡張された。CUDA の atomicAdd() 関数は、32-bit と 64-bit の整数および浮動小数点データに適用できる。すべての浮動小数点 atomic add operation は round-to-nearest-even で丸められる（従来の FP32 atomic addition は round-to-zero を使用）。

<span id="section-3-6"></span>

### 3.6 GP100 における L1/L2 キャッシュの変更

Fermi と Kepler の GPU は、ワークロードに応じて L1 と shared memory の間で容量を配分できる 64 KB の configurable shared memory and L1 cache を備えていたが、Maxwell から cache hierarchy が変更された。GP100 SM は専用の shared memory pool（64 KB/SM）と、ワークロードに応じて texture cache としても使える L1 cache を持つ。統合 L1/texture cache は memory access の coalescing buffer として機能し、warp のスレッドが要求したデータをまとめてから、その warp へ渡す。

> **注：** 1 つの CUDA Thread Block だけでは 64 KB の shared memory を割り当てられないが、2 つの Thread Block がそれぞれ 32 KB を使用することはできる、など。

SM ごとに専用 shared memory があるため、アプリケーションは最適性能を得るための L1/shared 分割を選ぶ必要がなくなり、SM 当たり 64 KB の全容量を常に shared memory として利用できる。GP100 は統合 4096 KB L2 cache を備え、GPU 全体で効率よく高速にデータを共有する。対して GK110 の L2 cache は 1536 KB、GM200 は 3072 KB だった。オンチップ cache が増えると GPU DRAM への要求が減り、ボード全体の消費電力と memory bandwidth の需要を抑え、性能を高められる。

<span id="section-3-7"></span>

### 3.7 GPUDirect の強化

膨大な地質データを扱う場合でも、複雑な科学問題の解決策を研究する場合でも、可能な限り高い data throughput と低い latency を提供するコンピューティングプラットフォームが必要である。GPUDirect は、1 台のコンピューター内の GPU、またはネットワーク上の異なるサーバーにある GPU が、CPU/system memory を経由せず直接データを交換できる機能である。

Kepler GK110 で GPUDirect に導入された RDMA 機能により、InfiniBand（IB）adapter、network interface card（NIC）、SSD などの第三者製デバイスは、同一システム内の複数 GPU のメモリへ直接アクセスできる。不要な memory copy を除き、CPU overhead を大幅に下げ、GPU memory との間で MPI message を送受信する latency を著しく短縮する。system memory bandwidth の需要も減らし、GPU DMA engine を他の CUDA task に解放する。

GP100 は、source GPU memory からデータを読み、PCIe 経由で target NIC memory へ書く実効 RDMA bandwidth を 2 倍にする。GPUDirect bandwidth の倍増は多くのユースケース、とりわけ Deep Learning に非常に重要である。実際、Deep Learning machine は GPU と CPU の比率が高く（CPU 当たり 8 基の GPU を持つ場合もある）、GPU がデータ転送を CPU に戻さず I/O とすばやくやり取りできることがきわめて重要になる。

<span id="section-3-8"></span>

### 3.8 Compute Capability

GP100 GPU は新しい Compute Capability 6.0 を支援する。[表 2](#table-02) は NVIDIA GPU アーキテクチャの異なる Compute Capability のパラメーターを比較する。

<span id="table-02"></span>

![Kepler GK110、Maxwell GM200、Pascal GP100 の Compute Capability](../../papers/nvidia-pascal-architecture/table-02.png)

**表 2。** Compute Capability：GK110、GM200、GP100 の比較

<span id="section-3-9"></span>

### 3.9 Tesla P100：HBM2 を搭載する世界初の GPU

近年、GPU によるコンピュートアプリケーションの高速化が大きく普及し、多くのアプリケーションでデータ需要も増大した。GPU で解かれる問題ははるかに大規模になり、より大きなデータセットとより高い DRAM bandwidth を必要としている。この raw bandwidth 需要に応えるため、Tesla P100 は High Bandwidth Memory 2（HBM2）を使用した最初の GPU アクセラレーターとなった。HBM2 は DRAM のパッケージ方法と GPU への接続方法を根本的に変え、DRAM bandwidth を大幅に高める。

従来の GDDR5 GPU board design のように GPU 周囲へ多数の discrete memory chip を配置する代わりに、HBM2 は複数の memory die を垂直に重ねた 1 個以上の stack を使う。memory die は through-silicon via と microbump で形成した微細配線によって接続される。1 枚の 8 Gb HBM2 die には 5,000 個を超える through-silicon via hole がある。次に passive silicon interposer を使って memory stack と GPU die を接続する。HBM2 stack、GPU die、Silicon interposer は単一の 55mm x 55mm BGA package に収められる。[図 9](#figure-09) は GP100 と 2 つの HBM2 stack の図、[図 10](#figure-10) は GPU と memory を含む実際の P100 の顕微鏡写真である。

<span id="figure-09"></span>

![GP100 に隣接する HBM2 stack の断面](../../papers/nvidia-pascal-architecture/figure-09.png)

**図 9。** GP100 に隣接する HBM2 stack の断面図

<span id="figure-10"></span>

![P100 HBM2 stack と GP100 GPU の断面顕微鏡写真](../../papers/nvidia-pascal-architecture/figure-10.png)

**図 10。** P100 HBM2 stack と GP100 GPU の断面顕微鏡写真

[図 10](#figure-10) の顕微鏡写真は Tesla P100 HBM2 stack と GP100 GPU の断面を示す。左上の HBM2 stack は 5 枚の die、すなわち base die とその上の 4 枚の memory die からなる。最上層の memory die は非常に厚い。組み立て時には最上部の die と GPU を同じ高さに研削し、ヒートシンク用に同一平面を作る。

前世代の HBM1 と比べ、HBM2 はより大きな memory capacity と memory bandwidth を提供する。HBM2 は stack 当たり 4 枚または 8 枚の DRAM die を支援するが、HBM1 は 4 枚だけだった。HBM2 は DRAM die 当たり最大 8 Gb を支援する一方、HBM1 は die 当たり 2 Gb に限られた。HBM1 の bandwidth は stack 当たり 125 GB/sec が上限だったが、P100 は HBM2 により stack 当たり 180GB/sec を支援する。

GP100 full-chip block diagram（[図 7](#figure-07)）に示すように、GP100 GPU は 4 基の HBM2 DRAM stack に接続する。各 HBM2 stack には 2 基の 512-bit memory controller が接続し、実効 4096-bit 幅の HBM2 memory interface を構成する。初期の Tesla P100 アクセラレーターは、4-die の HBM2 stack を 4 基、合計 16 GB の HBM2 memory を搭載して出荷される。

<span id="section-3-9-1"></span>

#### 3.9.1 メモリの耐障害性

HBM2 memory のもう 1 つの利点は、error correcting code（ECC）機能をネイティブに支援することである。ECC はデータ破損の影響を受けやすいコンピュートアプリケーションの信頼性を高める。GPU が非常に大きなデータセットを処理する、またはアプリケーションを長時間実行する大規模クラスターコンピューティング環境で特に重要である。

ECC technology は single-bit soft error がシステムに影響する前に検出し、訂正する。対して GDDR5 は memory 内容を保護する internal ECC を持たず、GDDR5 bus の error detection に限られる。memory controller や DRAM 自体の error は検出されない。

GK110 Kepler GPU は利用可能な memory の一部を explicit ECC storage に割り当て、GDDR5 を ECC で保護した。GDDR5 全体の 6.25% が ECC bit 用に予約される。たとえば 12 GB の Tesla K40 では、総 memory の 750 MB が ECC operation 用に予約され、ECC を有効にした場合の利用可能容量は 12 GB 中 11.25 GB だった。また ECC bit へのアクセスにより、典型的なワークロードの memory bandwidth は ECC なしの場合と比べて 12～15% 低下した。HBM2 は ECC をネイティブに支援するため、Tesla P100 にこの capacity overhead はなく、bandwidth penalty なしで ECC を常時有効にできる。GK110 GPU と同様、GP100 GPU の register file、shared memory、L1 cache、L2 cache、および Tesla P100 アクセラレーターの HBM2 DRAM は Single-Error Correct Double-Error Detect（SECDED）ECC code で保護される。

<span id="section-3-10"></span>

### 3.10 Tesla P100 の設計

Tesla P100 system architecture で最も興味深い新機能の 1 つは、GP100 GPU と HBM2 memory stack を搭載し、NVLink と PCIe の接続も提供する新しい board design である。1 基以上の P100 アクセラレーターを workstation、server、大規模 computing system で使用できる。P100 アクセラレーターは 140mm x 78mm で、GPU に必要な各種電圧を供給する高効率 voltage regulator を備える。P100 の定格は 300W である。

[図 11](#figure-11) に Tesla P100 アクセラレーターの前面、[図 12](#figure-12) に背面を示す。

<span id="figure-11"></span>

![Tesla P100 アクセラレーターの前面](../../papers/nvidia-pascal-architecture/figure-11.png)

**図 11。** Tesla P100 アクセラレーター（前面）

<span id="figure-12"></span>

![Tesla P100 アクセラレーターの背面](../../papers/nvidia-pascal-architecture/figure-12.png)

**図 12。** Tesla P100 アクセラレーター（背面）

<span id="section-4"></span>

## 4 NVLink 高速インターコネクト

NVLink は GPU アクセラレーテッドコンピューティング向けの NVIDIA の新しい高速インターコネクト技術である。現在 Tesla P100 accelerator board と Pascal GP100 GPU に実装され、GPU 間通信と GPU から system memory へのアクセスの両方を大幅に高速化する。

High Performance Computing cluster の node では複数の GPU が一般に使われる。現在では node 当たり最大 8 基の GPU が一般的であり、multiprocessing system では強力な interconnect がきわめて有益である。NVLink の構想は、PCI Express Gen 3（PCIe）よりはるかに広い bandwidth を GPU に提供し、GPU ISA と互換性を持たせて shared memory multiprocessing workload を支援する interconnect を作ることだった。

<span id="figure-13"></span>

![8 基の NVIDIA Tesla P100 GPU を搭載した NVIDIA DGX-1](../../papers/nvidia-pascal-architecture/figure-13.png)

**図 13。** 8 基の NVIDIA Tesla P100 GPU を搭載した NVIDIA DGX-1

NVLink-connected GPU では、プログラムは local memory だけでなく別の GPU に接続された memory 上でも直接実行でき、memory operation の正しさも保たれる（たとえば Pascal の atomic operation を完全に支援する）。

NVLink は NVIDIA の新しい High-Speed Signaling interconnect（NVHS）を使う。NVHS は最大 20 Gb/sec で動作する differential pair でデータを伝送する。8 本の differential connection が一方向にデータを送る Sub-Link を構成し、各方向に 1 本ずつの sub-link で 2 つの processor（GPU-GPU または GPU-CPU）を接続する Link を構成する。単一 Link は endpoint 間で最大 40 GB/sec の bidirectional bandwidth を支援する。複数の Link を Gang にまとめ、processor 間でさらに高い bandwidth を得ることもできる。Tesla P100 の NVLink 実装は最大 4 本の Link を支援し、aggregate maximum bidirectional bandwidth が 160 GB/sec の ganged configuration を可能にする。構成は[図 14](#figure-14)と[図 15](#figure-15)に示す。

<span id="section-4-1"></span>

### 4.1 NVLink の構成

多数の topology が可能であり、アプリケーションごとに異なる構成を最適化できる。本節では次の NVLink 構成を説明する。

- GPU-GPU NVLink 接続
- CPU-GPU NVLink 接続

<span id="section-4-1-1"></span>

#### 4.1.1 GPU-GPU NVLink 接続

[図 14](#figure-14) は、NVLink で完全接続された 2 組の 4-GPU quad、quad 間の NVLink 接続、各 quad 内の GPU からそれぞれの CPU への直接 PCIe 接続を持つ 8-GPU Hybrid Cube Mesh を示す。2 組の quad 間を別々の NVLink 接続でまたぐことにより、各 CPU への PCIe uplink の負荷を軽減し、system memory と inter-CPU link を介した転送も避けられる。

<span id="figure-14"></span>

![8-GPU Hybrid Cube Mesh アーキテクチャ](../../papers/nvidia-pascal-architecture/figure-14.png)

**図 14。** 8-GPU Hybrid Cube Mesh アーキテクチャ

8-GPU Hybrid Cube Mesh の各半分は shared memory multiprocessor として動作し、remote node も peer 経由の DMA で memory を共有できる。GPU-GPU traffic がすべて NVLINK を流れるため、PCIe は NIC（図示せず）への接続または system memory traffic へのアクセスに全容量を利用できる。この構成は汎用 Deep Learning アプリケーションに一般的に推奨され、NVIDIA の新しい DGX-1 server に実装される。

[図 15](#figure-15) は 4-GPU cluster を示し、各 GPU が単一の NVLink で他のすべての GPU に接続される。この場合、peer は双方向 40 GB/sec（double link では双方向 80GB/sec）で通信でき、GPU 間で堅牢にデータを共有できる。

<span id="figure-15"></span>

![4 基の GPU を NVLink で接続し CPU を PCIe で接続](../../papers/nvidia-pascal-architecture/figure-15.png)

**図 15。** NVLink で 4 基の GPU を接続し、CPU は PCIe で接続

<span id="section-4-1-2"></span>

#### 4.1.2 CPU-GPU NVLink 接続

NVLink は主として複数の NVIDIA Tesla P100 アクセラレーターを接続するが、CPU-GPU interconnect としても使用できる。たとえば Tesla P100 アクセラレーターは、NVIDIA NVLink technology を搭載する IBM POWER8 に接続できる。POWER8 with NVLink™ は 4 本の NVLink を支援する。

[図 16](#figure-16) は NVLink-enabled CPU に 1 基の GPU を接続した構成を示す。この場合、GPU は system memory に最大 160 GB/sec の bidirectional bandwidth、すなわち PCIe の 5 倍でアクセスできる。

<span id="figure-16"></span>

![1 基の GPU を 4 本の NVLink で CPU に接続](../../papers/nvidia-pascal-architecture/figure-16.png)

**図 16。** NVLink GPU-CPU インターコネクト

[図 17](#figure-17) は、CPU から各 GPU へ 2 本の NVLink を接続したシステムを示す。各 GPU に残る 2 本の link は peer-to-peer 通信に使う。

<span id="figure-17"></span>

![2 基の GPU と CPU を各 2 本の NVLink で接続](../../papers/nvidia-pascal-architecture/figure-17.png)

**図 17。** 2 基の GPU と CPU を 80 GB/sec の双方向帯域幅で接続

<span id="section-4-2"></span>

### 4.2 Tesla P100 への NVLink インターフェース

Tesla P100 Design 節で説明したとおり、P100 accelerator は NVLink interconnection を搭載する。P100 は 2 つの 400-pin high speed connector を持つ。一方は module 内外への NVLink signal、もう一方は電力、control signal、PCIe I/O の供給に使う。

Tesla P100 accelerator は、より大きな GPU carrier または system board に搭載できる。GPU carrier は他の P100 accelerator または PCIE controller に適切に接続する。P100 accelerator は従来の GPU board より小さいため、従来以上に多くの GPU を高密度に搭載した server を容易に構築できる。NVLink が追加 bandwidth を提供するため、GPU-GPU 通信が PCIe bandwidth の制約で bottleneck になることはなく、従来不可能だった GPU clustering が可能になる。

GPU architecture interface のレベルでは、NVLink controller は High-Speed Hub（HSHUB）という別の新ブロックを介して GPU 内部と通信する。HSHUB は GPU-wide crossbar と、High-Speed Copy Engine（HSCE）などの他の system element に直接アクセスする。HSCE は peak NVLink rate で GPU へデータを出し入れするために使える。[図 18](#figure-18) は NVLink と HSHUB、および GP100 GPU の上位 block との関係を示す。

<span id="figure-18"></span>

![GP100 における NVLink と HSHUB、crossbar、copy engine、memory controller の関係](../../papers/nvidia-pascal-architecture/figure-18.png)

**図 18。** GP100 の他の主要ブロックに対する NVLink の関係

詳細は[第 9 節](#section-9)を参照されたい。

<span id="section-5"></span>

## 5 Unified Memory

Unified Memory は CUDA programming model の重要な機能であり、システム内の CPU と GPU のすべての memory にアクセスする単一の統合 virtual address space を提供し、GPU programming とアプリケーションの GPU 移植を大幅に簡素化する。Pascal GP100 の新機能は Unified Memory の機能と性能を拡張し、GPU コンピューティングを大きく前進させる。

現代の processor で高性能を得る鍵は、hardware computational unit がデータへ高速かつ直接アクセスできるようにすることである。長年にわたり NVIDIA は GPU memory access と data sharing を継続的に改善し、GPU programmer が memory allocation や GPU-CPU 間 data transfer の管理よりも parallel application の構築に集中できるよう簡素化してきた。

長年、一般的な PC や cluster node では CPU と各 GPU の memory が物理的に分離され、通常は PCIe などの interconnect bus で隔てられていた。初期の CUDA では、GPU programmer が CPU/GPU memory allocation と data transfer を明示的に管理しなければならなかった。CPU と GPU が共有するデータには system memory と GPU memory の 2 つの allocation が必要だったため、これは難しかった。programmer は明示的な memory copy call を使い、最新のデータを両者間で移動する必要があった。適切な時点で適切な場所にデータを置く処理は application を複雑にし、新しい GPU programmer の学習曲線を急にした。

sparse memory access では、explicit data transfer が性能を損なう場合もある。たとえば CPU が数 byte をランダムに書き換えただけで配列全体を GPU に戻すと、transfer latency overhead が増える。memory transfer の管理、memory locality の改善、asynchronous memory copy などの手法は性能を高められるが、いずれもプログラミング上の注意をさらに必要とする。

<span id="section-5-1"></span>

### 5.1 Unified Memory の歴史

2009 年に導入された NVIDIA Fermi GPU architecture は、3 つの主要な GPU memory space（thread private local memory、thread block shared memory、global memory）にまたがる統合 GPU address space を実装した。この統合 address space は GPU memory addressing にのみ適用され、各空間に別々の命令と pointer を使う代わりに、単一の load/store 命令と pointer address で任意の GPU memory space（global、local、shared memory）へアクセスできるようにして、主にコンパイルを簡素化した。当時としては大きな進歩だった完全な C/C++ pointer 対応も可能になった。

2011 年、CUDA 4 は Unified Virtual Addressing（UVA）を導入し、CPU と GPU memory に単一の virtual memory address space を提供した。GPU code の pointer は、同一または別の GPU の memory、CPU memory、on-chip shared memory のいずれに置かれていてもアクセスできる。UVA は Zero-Copy memory、すなわち memcpy なしで GPU code から PCIe 経由で直接アクセスできる pinned CPU memory を可能にする。Zero-Copy は Unified Memory の利便性の一部を提供するが、常に低帯域・高 latency の PCIe 経由で GPU がアクセスするため、同等の性能は提供しない。

CUDA 6 は CPU と GPU で共有する managed memory pool を作り、CPU-GPU 間をつなぐ Unified Memory を導入した。managed memory は CPU と GPU の両方から単一 pointer でアクセスできる。CUDA system software は Unified Memory に割り当てたデータを GPU と CPU の間で自動移行し、CPU 上の code には CPU memory、GPU 上の code には GPU memory のように見せる。ただし CUDA 6 Unified Memory は Kepler と Maxwell GPU architecture の機能に制約された。CPU が触れた managed memory は、kernel launch 前にすべて GPU と同期する必要があった。CPU と GPU は managed memory allocation に同時アクセスできず、Unified Memory address space は GPU physical memory の容量に制限された。

<span id="figure-19"></span>

![CPU と GPU から CUDA 6 Unified Memory へのアクセス](../../papers/nvidia-pascal-architecture/figure-19.png)

**図 19。** CUDA 6 Unified Memory

[図 20](#figure-20) は、CUDA 6 の Unified Memory が単一の data pointer を提供し、明示的な CPU-GPU memory copy を要件ではなく最適化に変えることで、GPU への code porting を簡素化する例を示す。

<span id="figure-20"></span>

![CUDA 6 Unified Memory による GPU porting の簡素化を示す code 比較](../../papers/nvidia-pascal-architecture/figure-20.png)

**図 20。** CUDA 6 Unified Memory による GPU への code porting の簡素化

（CPU code と GPU code のどちらからもアクセスできる data pointer を返す、新しい managed memory allocator によって実現する。）

<span id="section-5-2"></span>

### 5.2 Pascal GP100 Unified Memory

Pascal GP100 は CUDA 6 Unified Memory の利点を拡張し、CPU と GPU 間の programming と memory sharing をさらに簡素化し、CPU parallel compute application を GPU へ容易に移植して大幅な高速化を得る機能を追加する。これらの改善を可能にする主な hardware feature は、large address space と page fault capability の 2 つである。

GP100 は GPU addressing capability を拡張し、49-bit（512 TB）virtual memory addressing を可能にする（GP100 は 47-bit、128 TB の physical memory addressing にも対応する）。これは現代の CPU の 48-bit virtual address space と GPU 自身の memory を覆うのに十分大きい。そのため GP100 Unified Memory program は、システム内のすべての CPU と GPU の完全な address space を単一の virtual address space として扱い、どの processor の physical memory size にも制限されずにアクセスできる（[図 21](#figure-21)）。

GP100 の memory page fault 対応は、よりシームレスな Unified Memory 機能を提供する重要な新機能である。system-wide virtual address space と組み合わせた page fault には複数の利点がある。まず、CUDA system software は kernel launch ごとにすべての managed memory allocation を GPU と同期する必要がなくなる。GPU 上の kernel が自身の memory に resident でない page へアクセスすると fault が発生し、その page を GPU memory へ on-demand で自動移行できる。別の方法として、page を GPU address space に map し、PCIe または NVLink interconnect 経由でアクセスできる（access 時の mapping が migration より速い場合もある）。Unified Memory は system-wide である点に注意されたい。GPU（および CPU）は CPU memory またはシステム内の別 GPU の memory から memory page を fault し、migrate できる。

<span id="figure-21"></span>

![CPU と GPU の physical memory にまたがる unified virtual memory](../../papers/nvidia-pascal-architecture/figure-21.png)

**図 21。** Pascal GP100 Unified Memory は GPU physical memory の容量に制限されない。

新しい page fault mechanism により、Unified Memory では global data coherency が保証される。GP100 では programmer による同期なしに CPU と GPU が Unified Memory allocation へアクセスできることを意味する。Kepler と Maxwell GPU では、GPU kernel の実行中に CPU が Unified Memory allocation へアクセスした場合の coherency を保証できず、これは禁止されていた。

> **注：** あらゆる parallel application と同様、processor 間の data hazard を避けるため、開発者は正しい同期を確保する必要がある。

最後に、対応 OS platform では default OS allocator（malloc や new など）で割り当てた memory に、GPU code と CPU code の両方から同じ pointer でアクセスできる（[図 22](#figure-22)）。このような system では Unified Memory を default にでき、special allocator や専用 managed memory pool の作成は不要である。さらに GP100 の大きな virtual address space と page fault capability により、application は system virtual memory 全体へアクセスできる。すなわち application は memory system を oversubscribe し、system の total physical capacity より大きな array を allocate、access、share して、非常に大きな dataset を out-of-core processing できる。

system allocator で Unified Memory を有効にするには OS に一定の変更が必要である。NVIDIA は Red Hat と協力し、Linux community 内でこの機能を実現する作業を進めている。

<span id="figure-22"></span>

![default system allocator を用いる Pascal Unified Memory](../../papers/nvidia-pascal-architecture/figure-22.png)

**図 22。** OS の支援により、Pascal は default system allocator で Unified Memory を支援できる。

（ここでは malloc だけで、システム内の任意の CPU または GPU からアクセスできる memory を割り当てられる。）

<span id="section-5-3"></span>

### 5.3 Unified Memory の利点

programmer が Unified Memory から得る主な利点は 2 つある。

- より単純な programming model と memory model。Unified Memory は explicit device memory management を要件ではなく最適化にすることで、GPU の parallel programming への参入障壁を下げる。programmer は device memory の allocation と copy の細部に煩わされず、parallel code の開発に集中できる。GPU programming の習得と既存 code の GPU 移植が容易になる。
- 初心者だけの機能ではない。Unified Memory は複雑な data structure と C++ class も GPU 上で大幅に扱いやすくする。default system allocator で Unified Memory を支援する system では、hierarchical または nested data structure に system 内の任意の processor から自動アクセスできる。GP100 では system の total memory size より大きな dataset を out-of-core で扱える。
- data locality による性能。Unified Memory は CPU と GPU の間で on demand にデータを移行し、global shared data の使いやすさを提供しながら、GPU 上の local data と同等の性能を実現できる。この機能の複雑さは CUDA driver と runtime の内部に隠され、application code を簡単に記述できる。migration の目的は各 processor の full bandwidth を得ることにあり、高い HBM2 memory bandwidth は GP100 GPU の compute throughput を満たすために不可欠である。GP100 の page fault により、CPU または GPU がアクセスする page を事前に予測できない sparse data access の program や、CPU と GPU が同じ array allocation の一部へ同時アクセスする program でも locality を確保できる。

重要なのは、必要な場合に data management と CPU-GPU concurrency を明示的に最適化する道具が CUDA programmer に残されていることである。CUDA 8 は memory usage hint と explicit prefetching を runtime に与える有用な API を導入する。これらの道具は explicit GPU memory allocation の制約へ戻ることなく、explicit memory copy と pinning API と同じ機能を提供する。

<span id="section-6"></span>

## 6 Compute Preemption

Pascal GP100 の新しい Compute Preemption は、GPU 上で実行中の compute task を命令単位で中断し、その context を GPU DRAM へ swap out できる。他の application を swap in して実行した後、元の task の context を再び swap in し、中断位置から実行を続けられる。

Compute Preemption は、長時間実行または不適切に動作する application が system を独占し、task の完了待ちで system が応答しなくなり、task が timeout または OS/CUDA driver に kill される重要な問題を解決する。Pascal より前は、compute task と display task を同じ GPU で動かす system で long-running compute kernel が OS と他の visual application を応答不能・操作不能にし、kernel の timeout までその状態が続く場合があった。このため programmer は compute-only GPU を別に搭載するか、従来 GPU の制約に合わせて application を慎重に記述し、workload を短い execution timeslice に分割して timeout や OS による終了を避ける必要があった。

実際に多くの application が long-running process を必要とする。GP100 の Compute Preemption により、こうした application は大規模 dataset の処理や特定条件の成立待ちに必要なだけ実行でき、visual application は滑らかで interactive な状態を保つ。programmer が code を短い timeslice で動かすために苦労する必要もない。

Compute Preemption は single-GPU system で compute kernel を interactive に debug することも可能にする。developer productivity に重要な機能である。対して Kepler GPU architecture が提供したのは、compute kernel の thread block 単位という粗い preemption だけだった。この block-level preemption では、hardware が別 context へ switch する前に thread block 内の全 thread を完了しなければならない。しかし debugger の使用中、thread block 内の命令で GPU breakpoint に達すると thread block は未完了のため、block-level preemption ができなかった。Kepler と Maxwell でも compile 時の instrumentation により debugger の中核機能を提供できたが、GP100 はより堅牢で軽量な debugger implementation を支援できる。

<span id="section-7"></span>

## 7 NVIDIA DGX-1 Deep Learning スーパーコンピューター

data scientist と AI researcher は Deep Learning system に accuracy、simplicity、speed を求める。training と iteration が速ければ、最終的に innovation と time to market も速くなる。NVIDIA DGX-1 は完全統合した hardware と software を備え、迅速かつ容易に導入できる、世界初の Deep Learning 専用 server である。最大 170 FP16 TFLOPS という革新的な性能によって training time を大幅に短縮し、NVIDIA DGX-1 は「1 台の箱に収めた」最初の AI スーパーコンピューターとなる。

NVIDIA DGX-1 server は、Tesla P100 accelerator を NVLink で相互接続した最初の server である。8 基の Tesla P100 accelerator 構成を提供し、standalone use または cluster integration 向けに、高性能・高信頼性 component を 3U rack-mount chassis に搭載する。

8-GPU 構成は、NVLink で完全接続した 2 組の P100 GPU quad を持ち、[図 14](#figure-14)の Hybrid Cube Mesh topology のように追加の 4 本の NVLink で quad 同士をつなぐ。quad 内のすべての GPU は PCIe で CPU 接続の PCIe switch にも直接接続する。

Deep Learning 向けに調整した software と強力な hardware を組み合わせた NVIDIA DGX-1（[図 23](#figure-23)）は、developer と researcher に高性能 GPU-accelerated Deep Learning application の開発、test、network training の turnkey solution を提供する。

<span id="figure-23"></span>

![NVIDIA DGX-1 server](../../papers/nvidia-pascal-architecture/figure-23.png)

**図 23。** NVIDIA DGX-1 Server

<span id="section-7-1"></span>

### 7.1 1 台の箱に 250 台のサーバー

[表 3](#table-03) は Dual Xeon system と DGX-1 server の Alexnet training time を比較する。見てわかるように、DGX-1 の raw processing power は raw TFLOPS と aggregate node bandwidth の両方で Dual Xeon を大きく上回る。Dual Xeon は Alexnet を 2 時間の turn-around-time（TAT）で訓練するために 250 node 以上を必要とするが、DGX-1 は 1 node だけでよい。

<span id="table-03"></span>

![Pascal GP100 と Xeon system の Alexnet training time](../../papers/nvidia-pascal-architecture/table-03.png)

**表 3。** Alexnet Training Time：Pascal GP100 と Xeon の比較

<span id="section-7-2"></span>

### 7.2 1 年で DNN を 12 倍高速化

[図 24](#figure-24) は Alexnet における Pascal DGX-1 と Maxwell の 1 年間の DNN speedup を比較する。

<span id="figure-24"></span>

![4 基の Maxwell GPU と 8 基の Pascal GPU による Alexnet training time](../../papers/nvidia-pascal-architecture/figure-24.png)

**図 24。** NVIDIA の前回 GTC event 以降における Pascal DGX-1 と Maxwell の 1 年間の DNN speedup

<span id="section-7-3"></span>

### 7.3 DGX-1 Software Features

DGX-1 Base OS software は最小限の労力で Deep Learning をすぐ始められるようにする。GPU 向けに調整した industry-standard Linux distribution に基づく DGX-1 software stack は、CUDA 8.0 と NVIDIA Deep Learning SDK の最新版を含み、Deep Learning application が Tesla P100 の高性能機能を利用して主要な Deep Learning framework とその application をすべて高速化できる。

<span id="section-7-4"></span>

### 7.4 NVIDIA DGX-1 System Specifications

NVIDIA DGX-1 は、迅速かつ容易に導入できる完全統合 hardware/software を備えた世界初の Deep Learning 専用 server である。その革新的な性能は training time を大幅に短縮し、NVIDIA DGX-1 を「1 台の箱に収めた」最初の AI スーパーコンピューターにする。[表 4](#table-04) に NVIDIA DGX-1 の system specifications を示す。

<span id="table-04"></span>

![NVIDIA DGX-1 system specifications](../../papers/nvidia-pascal-architecture/table-04.png)

**表 4。** NVIDIA DGX-1 System Specifications

<span id="section-8"></span>

## 8 結論

Pascal architecture で構築した NVIDIA の新しい NVIDIA Tesla P100 GPU accelerator は、従来は解けなかった問題を顧客が計算できるようにする breakthrough を結集する。NVIDIA Tesla P100 は上から下まで、compute performance、memory bandwidth、capacity、connectivity、power efficiency という驚くべき innovation を備え、次世代 HPC/AI system の computational engine に必要な能力を提供する。

<span id="section-9"></span>

## 9 付録 A：NVLink の信号・プロトコル技術

NVLink は NVIDIA の High Speed Signaling technology（NVHS）を使う。データは signal pair 当たり 20 Gbit/sec で differential transmission される。各方向の 8 本の differential pair を組み合わせて 1 本の link とし、これが基本 building block になる。単一 link の raw bidirectional bandwidth は 40 GB/sec である。信号方式は NRZ（Non-Return-to-Zero）。link は DC-coupled で、differential impedance は 85 Ohms である。link は polarity inversion と lane reversal に耐え、効率的な PCB routing を可能にする。die 上では 1.25GHz の data rate で、128-bit Flit（Flow control digit）を使い、PHY（physical level circuit）から NVLink controller へデータを送る。NVHS は embedded clock を使う。receiver は recovered clock で incoming data を取り込む。

<span id="section-9-1"></span>

### 9.1 NVLink Controller Layers

NVLink controller は Physical Layer（PL）、Data Link Layer（DL）、Transaction Layer（TL）の 3 layer からなる。protocol は variable-length packet を使い、packet size は 1 flit（たとえば単純な read request command）から 18 flit（address extension を伴う 256B data transfer の write request with data）までである。[図 25](#figure-25) に NVLink の layer と link、すなわち Physical Layer（PHY）、Data Link Layer（DL）、Transaction Layer（TL）を示す。

<span id="section-9-1-1"></span>

#### 9.1.1 Physical Layer（PL）

PL は PHY と接続する。PL は deskew（8 lane 全体）、framing（各 packet の開始位置の判定）、scrambling/descrambling（clock recovery に必要な bit transition density の確保）、polarity inversion、lane reversal、受信 data の Data Link Layer への配送を担当する。[図 25](#figure-25) に NVLink の layer と link、すなわち Physical Layer（PHY）、Data Link Layer（DL）、Transaction Layer（TL）を示す。

<span id="figure-25"></span>

![4 本の link にまたがる NVLink Physical Layer、Data Link Layer、Transaction Layer](../../papers/nvidia-pascal-architecture/figure-25.png)

**図 25。** NVLink の Layer と Link：Physical Layer（PHY）、Data Link Layer（DL）、Transaction Layer（TL）

<span id="section-9-1-2"></span>

#### 9.1.2 Data Link Layer（DL）

Data Link Layer は主に link 上での packet の reliable transmission を担当する。送信 packet は 25-bit CRC（Cyclic Redundancy Check）で保護される。送信した packet は、link の反対側の receiver から positive acknowledgement（ACK）を受けるまで replay buffer に格納される。DL が incoming packet に CRC error を検出すると ACK を送らず、retransmitted data の受信を準備する。一方 transmitter は ACK がないため timeout し、replay buffer から data retransmission を開始する。packet は acknowledgement を受けた場合にのみ replay buffer から retire される。25-bit CRC は最大 5 個の random bit error、または任意の lane における最大 25-bit の burst error を検出できる。CRC は current header と previous payload（存在する場合）に対して計算する。

DL は link bring-up と maintenance も担当する。DL は Transaction Layer（TL）へデータを送る。

<span id="section-9-1-3"></span>

#### 9.1.3 Transaction Layer

Transaction Layer は synchronization、link flow control、virtual channel を扱い、複数の link を aggregate して processor 間に非常に高い communication bandwidth を提供できる。

<span id="section-10"></span>

## 10 付録 B：GPU による Deep Learning と Artificial Intelligence の高速化

コンピューティングの究極の目標は Artificial Intelligence、すなわち明示的な指示なしに自ら学べるほど知的な machine を構築することである。Deep Learning は現代の AI を実現するうえで不可欠な要素である。

Deep Learning により AI brain は周囲の世界を認識し、machine は学習して最終的に自ら意思決定する。その訓練には膨大なデータが必要であり、全データの処理には高度で複雑な deep neural network も必要となる。2012 年、Google の Deep Learning project である Google Brain は YouTube の動画を見て猫を認識することを学んだ。ただし、これには Google の data center で給電・冷却する 2,000 基の CPU（16,000 CPU core）が必要だった。この規模の machine を持つ組織は少ない。同じ頃、NVIDIA Research は Stanford University と共同し、GPU を Deep Learning に使用した。その結果、12 基の NVIDIA GPU で 2,000 基の CPU に相当する deep-learning performance を提供できた。

多くの人は、Krizhevsky、Sutskever、Hinton が現在「AlexNet」と呼ばれる convolutional neural network を用いて参加した 2012 年の ImageNet competition を Deep Learning 革命の始まりとみなしている。この network は GPU の parallel processing performance を使い、従来の computer vision competition 全体を大差で上回った。Artificial Intelligence と Deep Learning の歴史における画期的な出来事だった。Krizhevsky と team は computer vision code を一切書かなかった。代わりに、computer は Deep Learning を使って自ら image recognition を学んだ。彼らは neural network（AlexNet）を設計し、NVIDIA GPU 上で数兆回の数学演算を必要とする 100 万枚の example image で訓練した。Krizhevksy の AlexNet は人間が書いた最高の software を破った。

その後、GPU 上で動く deep neural network（DNN）は、特に computer vision、そして広くは machine perception に関するさまざまな algorithm domain を一つずつ制覇した。self-driving car から drug development の高速化、online image database の automatic image captioning、video chat application の smart real-time language translation まで、潜在的なユースケースは無限にある。machine が人間の世界と関わるあらゆる場所で、Deep Learning は魅力的な機会をもたらす。今日、deep neural network の利用には GPU が欠かせない。あらゆる Deep Learning user が CPU から 1 基以上の massively parallel GPU accelerator へ移行し、training time を劇的に短縮している。

<span id="section-10-1"></span>

### 10.1 Deep Learning の概要

Deep Learning は人間の脳の neural learning process を model 化する技法であり、継続的に学習し、賢くなり、時間とともにより正確な結果をより速く出す。子供は最初、大人からさまざまな形を正しく識別・分類するよう教えられ、やがて指導なしに形を識別できるようになる。同様に、Deep Learning または neural learning system は、basic object や occluded object などを効率よく識別し、同時に object に context を与えられるようになるため、object recognition と classification の訓練を受ける必要がある。

最も単純な水準では、人間の脳の neuron は入力されたさまざまな input を見て、それぞれに importance level を割り当て、他の neuron が処理するために output を渡す。

[図 26](#figure-26)の Perceptron は neural network の最も基本的な model で、人間の脳の neuron に似ている。図のとおり、Perceptron には認識・分類を学習する object のさまざまな feature を表す複数の input があり、各 feature には object の形状を定義する重要度に応じた weight が割り当てられる。

<span id="figure-26"></span>

![Perceptron の input、weight、sum、activation function、output](../../papers/nvidia-pascal-architecture/figure-26.png)

**図 26。** Perceptron は Neural Network の最も単純な Model

手書き数字 0 の識別を学習する Perceptron を考える。数字 0 は筆跡に応じてさまざまに書ける。Perceptron は数字 0 の image を複数の section に分解し、それぞれを feature x1～x4 に割り当てる。0 の右上の曲線を x1、下側の曲線を x2 に割り当てる、といった具合である。特定 feature に対応する weight は、その feature が手書き数字を 0 と正しく判断するうえでどの程度重要かを決める。diagram 中央の緑色の部分では、Perceptron が image の全 feature の weighted sum を計算し、その数字が 0 かどうかを判定する。続いてこの結果に function を適用し、数字が 0 かどうかを true または false で出力する。

neural network の重要な点は、より良い prediction を行うよう network を訓練することにある。手書き数字 0 を検出する Perceptron model（[図 26](#figure-26)）は、最初に 0 を定義する各 feature へ一連の weight を割り当てて訓練する。次に Perceptron へ数字 0 を与え、正しく識別するか確認する。data が network を通り、数字が 0 かどうかの結論に至るまでの流れが *forward propagation* phase である。neural network が数字を正しく識別しない場合、誤認識の理由と error の大きさを把握し、Perceptron が 0 を正しく識別するまで各 feature の weight を調整する必要がある。さまざまな筆跡の 0 を正しく識別するまで weight をさらに調整する必要がある。error を feed back し、0 を定義する各 feature の weight を調整するこの process を *backward propagation* と呼ぶ。diagram の equation は複雑に見えるが、説明した training process を数学的に表したものである。

Perceptron は非常に単純な neural network model だが、同様の概念に基づく高度な multi-layered neural network が広く使われている。network を object の正確な認識と分類ができるよう訓練すると、現場へ配備して繰り返し *inference* computation を実行する。inference（DNN が与えられた input から有用な情報を抽出する process）の例には、ATM に入金した小切手の手書き数字の識別、Facebook 写真の友人の識別、5,000 万人を超える Netflix user への映画推薦、自動運転車における車種・歩行者・道路上の危険の識別と分類、人間の音声の real-time translation がある。

[図 27](#figure-27)の multi-layered neural network model は、相互接続した複数の複雑な Perceptron-like node からなり、各 node が多数の input feature を見て、次の数 layer の相互接続 node へ output を送る場合がある。

<span id="figure-27"></span>

![Audi A7 を認識する multi-layer neural network が学習した階層的 feature](../../papers/nvidia-pascal-architecture/figure-27.png)

**図 27。** 複雑な Multi-layer Neural Network Model にはより多くの Compute Power が必要

[図 27](#figure-27)の model では、neural model の第 1 layer が image を複数の section に分けて線や角などの basic pattern を探し、第 2 layer が線を組み合わせて wheel、windshield、mirror などの higher-level pattern を探し、次の layer が vehicle type を識別し、最後の数 layer が特定 brand の model（ここでは Audi A7）を識別する。

fully connected layer の代わりに convolutional layer を使うこともできる。convolutional layer の neuron は、その下の layer の小さな領域にある neuron だけに接続する。通常、この領域は 5×5 neuron の grid（7×7 または 11×11 の場合もある）である。この grid の大きさを filter size と呼ぶ。したがって convolutional layer は input に convolution を実行すると考えられる。この connection pattern は retinal ganglion cell や primary visual cortex の cell など、脳の perceptual area に見られる pattern を模倣する。

DNN convolutional layer では、その layer の各 neuron で filter weight が同じである。通常、convolutional layer は異なる filter を持つ多数の「sub layer」として実装する。1 つの convolutional layer で数百種類の filter を使う場合がある。DNN convolutional layer は input に数百種類の convolution を同時に実行し、その結果を 1 つ上の layer から利用できるようにするものと考えられる。convolutional layer を含む DNN を Convolutional Neural Network（CNN）と呼ぶ。

<span id="section-10-2"></span>

### 10.2 NVIDIA GPU：Deep Learning の Engine

state-of-the-art DNN と CNN は、back-propagation で調整する parameter を数百万から 10 億以上持つ場合がある。さらに DNN が高精度を得るには大量の training data が必要であり、数十万から数百万の input sample を forward pass と backward pass の両方に通す必要がある。

従来型の CPU-based platform と比べて速度と energy efficiency の両面で優れるため、GPU が deep neural network の訓練における state of the art であることは、現在では academia と industry の双方で広く認められている。neural network は多数の同一 neuron から作られるため、本質的に高度に並列である。この parallelism は GPU に自然に対応し、CPU-only training を大幅に高速化する。

neural network は matrix math operation に強く依存し、複雑な multi-layered network は効率と速度の両面で膨大な floating point performance と bandwidth を必要とする。数千の processing core を持ち、matrix math operation 向けに最適化され、数十から数百 TFLOPS の性能を提供する GPU は、deep neural network-based AI/ML application に当然適した computing platform である。

NVIDIA はこの GPU-driven DNN/Artificial Intelligence（AI）革命の最前線にある。NVIDIA GPU は各種 application の DNN を 10～20 倍高速化し、training time を数週間から数日へ短縮する。この分野の expert と協力し、GPU design、system architecture、compiler、algorithm を継続的に改善している。過去 3 年間で、NVIDIA GPU-based computing platform は Deep Learning network training を 50 倍高速化した。

<span id="section-10-3"></span>

### 10.3 Tesla P100：Deep Neural Network 訓練向け最速 Accelerator

NVIDIA の最新かつ最先端の Pascal GPU architecture は、deep neural network training の性能を一桁高め、training time を大幅に短縮する。Tesla P100 は 3584 基の processing core により、Deep Learning application へ 21 TFLOPS を超える FP16 processing power を提供する。8 基の Tesla P100 accelerator を高速 NVLink interconnect で接続すると、きわめて複雑な multi-layered DNN の訓練に利用できる性能は 170 TFLOPS/sec へ大幅に増える

HBM2 memory、Unified Memory、高速 NVLink interconnect、より大きな cache、低 latency といった主要な architecture advance に加え、Tesla P100 は Deep Learning の性能を高める機能も持つ。Maxwell GPU architecture で初めて導入された 16-bit storage と arithmetic への対応を、Pascal GP100 GPU も備える。16-bit floating-point（FP16）storage/arithmetic 対応は neural network algorithm の性能をさらに高め、inference time を短縮した。

<span id="section-10-4"></span>

### 10.4 包括的な Deep Learning Software Development Kit

AI innovation は猛烈な速度で進んでいる。programming の容易さと developer productivity が最重要である。NVIDIA CUDA platform の programmability と豊富さにより、researcher は迅速に innovation を起こせる。NVIDIA は Deep Learning Software Development Kit（SDK）として NVDIA DIGITS™、cuDNN、cuBLAS などの高性能 tool/library を提供し、cloud、data center、workstation、embedded platform 上の革新的な GPU-accelerated ML application を支える。developer はどこでも作り、どこにでも配備したい。NVIDIA GPU は世界中で入手でき、あらゆる PC OEM が desktop、notebook、server、supercomputer 向けに提供し、Amazon、Google、IBM、Facebook、Baidu、Microsoft などの大企業が cloud でも提供する。Internet company、research、startup のあらゆる主要 AI development framework が NVIDIA GPU で高速化されている。どの AI development system を選んでも、GPU acceleration により速くなる。DNN があらゆる intelligent machine を駆動できるよう、ほぼすべての computing form-factor に対応する GPU も作った。GeForce は PC、Tesla は cloud と supercomputer、Jetson は robot と drone、DRIVE PX は car 向けである。すべてが同じ architecture を共有し、Deep Learning を高速化する（[図 28](#figure-28)）。

<span id="figure-28"></span>

![NVIDIA GPU で高速化する Deep Learning framework と platform](../../papers/nvidia-pascal-architecture/figure-28.png)

**図 28。** 高速化された Framework

<span id="section-10-5"></span>

### 10.5 NVIDIA GPU と DNN による Big Data 問題の解決

Baidu、Google、Facebook、Microsoft は、Deep Learning と AI processing に NVIDIA GPU を早期採用した企業だった。実際、AI technology によって、これらの企業は話し言葉に応答し、音声や text を別の言語へ翻訳し、image を認識して自動 tag 付けし、各 user に合わせた newsfeed、entertainment、product を推薦できる。startup と既存企業は、AI で新しい product/service を作り、operation を改善しようと競っている。わずか 2 年で NVIDIA が Deep Learning で協力する企業は約 35 倍、3,400 社以上へ増えた（[図 29](#figure-29)）。healthcare、life science、energy、financial service、automotive、manufacturing、entertainment などの産業は、膨大なデータから insight を推論することで恩恵を受ける。Facebook、Google、Microsoft が Deep Learning platform を誰でも使えるよう公開することで、AI-powered application は急速に広がる。この傾向を受け、Wired は最近 GPU の台頭を宣言した。

<span id="figure-29"></span>

![2013、2014、2015 年に NVIDIA と Deep Learning に取り組んだ組織](../../papers/nvidia-pascal-architecture/figure-29.png)

**図 29。** NVIDIA と Deep Learning に取り組む組織

<span id="section-10-5-1"></span>

#### 10.5.1 Self-driving Cars

人間を超える co-pilot で人を補助するにせよ、personal mobility service を変革するにせよ、都市の広大な駐車場需要を減らすにせよ、self-driving car は驚くべき社会的利益をもたらす可能性がある。運転は複雑である。予期しないことが起きる。着氷性の雨が道路をスケートリンクに変える。目的地への道路が閉鎖される。子供が車の前に飛び出す。self-driving car が遭遇し得るあらゆる状況を予測する software は書けない。そこに Deep Learning の価値があり、学習、適応、改善できる。NVIDIA DRIVE PX、NVIDIA DriveWorks、NVIDIA DriveNet（[図 30](#figure-30)）を使い、training system から in-car AI computer までを含む end-to-end Deep Learning platform を self-driving car 向けに構築している。結果は非常に刺激的である。人間を超える computer co-pilot と driverless shuttle の未来は、もはや science fiction ではない。

<span id="figure-30"></span>

![通常路面と雪道における NVIDIA DriveNet の環境認識](../../papers/nvidia-pascal-architecture/figure-30.png)

**図 30。** NVIDIA DriveNet

<span id="section-10-5-2"></span>

#### 10.5.2 Robots

大手製造 robot maker の FANUC は最近、箱から向きがランダムな物体を取り出すことを学習した assembly-line robot を実演した。GPU-powered robot は trial and error で学習した。この Deep Learning technology は Preferred Networks が開発し、同社は最近 The Wall Street Journal の Japan Seeks Tech Revival with Artificial Intelligence という記事で紹介された。

<span id="section-10-5-3"></span>

#### 10.5.3 Healthcare and Life Sciences

Deep Genomics は GPU-based Deep Learning を適用し、genetic variation がどのように disease へつながるかを理解しようとしている。Arterys は GPU-powered Deep Learning を使って medical image analysis を高速化する。その technology は GE Healthcare MRI machine に配備され、heart disease の診断を支援する。Enlitic は Deep Learning で medical image を解析し、tumor、ほとんど見えない fracture、その他の medical condition を識別している。

以上は GPU と DNN がさまざまな分野の Artificial Intelligence と machine learning を変革する例のほんの一部である。同様の例は文字どおり数千もある。

Deep Learning の breakthrough は多くの水準で AI capability を加速し、GPU-accelerated Deep Learning/AI system と algorithm はこの分野の exponential progress を可能にしている。

<span id="section-11"></span>

## 11 Notice

本ホワイトペーパーで提供するすべての情報（commentary、opinion、NVIDIA design specification、reference board、file、drawing、diagnostic、list、その他の document。総称および個別に「Materials」）は「現状のまま」提供される。NVIDIA は Materials に関し、明示、黙示、法定その他のいかなる warranty も行わず、noninfringement、merchantability、fitness for a particular purpose の黙示 warranty を明示的に否認する。

NVIDIA は予告なく、いつでも本 specification の correction、modification、enhancement、improvement、その他の change を行い、または product/service を中止する権利を留保する。customer は発注前に最新の関連 specification を入手し、情報が最新かつ完全であることを確認する必要がある。

NVIDIA product は、NVIDIA の authorized representative と customer が署名した個別の sales agreement に別段の定めがない限り、order acknowledgement 時に提示する NVIDIA standard terms and conditions of sale に従って販売される。本 specification の NVIDIA product 購入に customer general terms and conditions を適用することに、NVIDIA は明確に反対する。

NVIDIA product は medical、military、aircraft、space、life support equipment、または product の failure/malfunction が personal injury、death、property/environmental damage を合理的に引き起こし得る application での利用を想定して設計、許可、保証されていない。NVIDIA はそのような equipment/application への product の組み込みや使用について責任を負わず、組み込みや使用は customer 自身の責任で行う。

NVIDIA は、これらの specification に基づく product が further testing/modification なしに特定用途へ適することを表明または保証しない。各 product のすべての parameter を必ずしも test していない。計画する application に product が適合することを確認し、application failure や product failure を避けるために必要な test を行う責任は customer のみにある。customer の product design の弱点は NVIDIA product の quality/reliability に影響し、本 specification に含まれない追加または異なる condition/requirement を生じさせる場合がある。NVIDIA は（i）本 specification に反する方法で NVIDIA product を使用したこと、または（ii）customer product design に基づく、もしくは起因する failure、damage、cost、problem に責任を負わない。

本 specification は NVIDIA patent right、copyright、その他の NVIDIA intellectual property right に基づく license を明示または黙示に付与しない。NVIDIA が公開する third-party product/service の情報は、NVIDIA からそれらを使用する license、warranty、endorsement を与えるものではない。そのような情報の使用には、third party の patent right またはその他の intellectual property right に基づく third party license、もしくは NVIDIA の patent right またはその他の intellectual property right に基づく NVIDIA license が必要な場合がある。本 specification の情報は、NVIDIA が書面で複製を承認し、変更せず、関連するすべての condition、limitation、notice を伴う場合にのみ複製できる。

すべての NVIDIA design specification、reference board、file、drawing、diagnostic、list、その他の document（総称および個別に「Materials」）は「現状のまま」提供される。NVIDIA は Materials に関し、明示、黙示、法定その他のいかなる warranty も行わず、noninfringement、merchantability、fitness for a particular purpose の黙示 warranty を明示的に否認する。customer が何らかの理由で損害を被った場合でも、本稿で説明する product に関する NVIDIA の aggregate and cumulative liability は、NVIDIA の product terms and conditions of sale に従って制限される。

<span id="section-11-1"></span>

### 11.1 VESA DisplayPort

DisplayPort、DisplayPort Compliance Logo、DisplayPort Compliance Logo for Dual-mode Sources、DisplayPort Compliance Logo for Active Cables は、米国その他の国において Video Electronics Standards Association が所有する trademark である。

<span id="section-11-2"></span>

### 11.2 HDMI

HDMI、HDMI logo、High-Definition Multimedia Interface は HDMI Licensing LLC の trademark または registered trademark である。

<span id="section-11-3"></span>

### 11.3 ARM

ARM、AMBA、ARM Powered は ARM Limited の registered trademark である。Cortex、MPCore、Mali は ARM Limited の trademark である。その他の brand/product name は各所有者の財産である。「ARM」は ARM Holdings plc、その operating company である ARM Limited、および regional subsidiary である ARM Inc.、ARM KK、ARM Korea Limited、ARM Taiwan Limited、ARM France SAS、ARM Consulting（Shanghai）Co. Ltd.、ARM Germany GmbH、ARM Embedded Technologies Pvt. Ltd.、ARM Norway, AS、ARM Sweden AB を指す。

<span id="section-11-4"></span>

### 11.4 OpenCL

OpenCL は Apple Inc. の trademark であり、Khronos Group Inc. が license を受けて使用する。

<span id="section-11-5"></span>

### 11.5 Trademarks

NVIDIA、NVIDIA logo、CUDA、FERMI、KEPLER、MAXWELL、PASCAL、TITAN、Tesla、GeForce、NVIDIA DRIVE PX、NVIDIA DriveWorks、NVIDIA DriveNet、NVLink は、米国その他の国における NVIDIA Corporation の trademark および／または registered trademark である。その他の company/product name は、関連する各社の trademark である場合がある。

<span id="section-11-6"></span>

### 11.6 Copyright

© 2017 NVIDIA Corporation。無断転載を禁ず。

正確な印刷レイアウトと参考文献については原 PDF を正本とする。
