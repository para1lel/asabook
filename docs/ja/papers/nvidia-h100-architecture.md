---
title: 'NVIDIA H100 Tensor Core GPU Architecture'
createTime: 2026/09/09 14:37:38
permalink: /ja/papers/nvidia-h100-architecture/
pageClass: paper-reading
---

> [NVIDIA Corporation](https://www.nvidia.com/en-us/about-nvidia/)。*[NVIDIA H100 Tensor Core GPU Architecture](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c)*、V1.04、2023 年 5 月公開。<a href="/paper/nvidia-h100-architecture.pdf" target="_blank" rel="noopener noreferrer">原 PDF</a>。arXiv レコードおよび TeX ソースは存在しないため、正確な文言、印刷レイアウト、参考文献については公開 PDF を正本とする。GPU／メモリの最終クロックと最終的な TFLOPS 性能仕様を含む。

<span id="section-1"></span>

## 1 はじめに

NVIDIA® のアクセラレーテッドコンピューティング技術は、通常のコンピューターの能力をはるかに超える計算上の課題に取り組む。アクセラレーテッドコンピューティングに必要なのは、強力な GPU だけではない。NVIDIA® CUDA® 汎用プログラマブル GPU と、GPU で高速化された多数の SDK、API、アルゴリズムを組み合わせることで、複数の分野にわたり驚異的なアプリケーション高速化を実現するフルスタックのコンピューティングソリューションが得られる。分散 GPU コンピューティングシステムとソフトウェアは、データセンター全体に処理を拡張する。世界中のクラウドデータセンターでは NVIDIA GPU アクセラレーテッドシステムおよびアーキテクチャによるスケールアップとスケールアウトが進み、多様な AI、HPC、データ分析アプリケーションが実行されている。

15 年以上前、NVIDIA は G80 GPU とともに CUDA 並列コンピューティングプラットフォームを導入した。それ以来、CUDA のツールとライブラリは 3,000 万回以上ダウンロードされ、約 300 万人の開発者に利用されてきた。CUDA プラットフォームは、より強力な CUDA 対応 GPU、新しく多様な GPU アクセラレーテッドライブラリ群、ワークステーション、サーバー、アプリケーションによって継続的に改良、最適化、拡張され、NVIDIA アクセラレーテッドコンピューティングの適用範囲を広げてきた。

現在、NVIDIA はさまざまな産業、科学分野、アプリケーション向けのフルスタックソリューションを提供している。450 を超える NVIDIA の SDK、ツールキット、ライブラリ、モデルが、ゲームやデザインから、生命科学・地球科学、ロボティクス、自動運転車、量子コンピューティング、サプライチェーン物流、サイバーセキュリティ、5G、気候科学、デジタルバイオロジーなどに至る産業とアプリケーションに対応している。現在、25,000 社を超える企業が NVIDIA の AI 技術を利用している。

NVIDIA の CUDA プラットフォームはプログラミングが容易で機能も豊富なため、設計者、研究者、エンジニアは迅速にイノベーションを起こせる。また、プラットフォームソフトウェアが継続的に最適化されるため、NVIDIA 製品のライフサイクルを通じて数倍の高速化を体験することも珍しくない。

NVIDIA GPU は世界最大級のデータセンターの多くで利用され、AI、HPC、データ分析のシステムとアプリケーションを大幅に高速化している。クラウドデータセンターでは、NVIDIA GPU によって AI トレーニングが急速にスケールアップされ、推論アプリケーションがスケールアウトされている。現在では、幅広い企業利用に向けて多様な種類の AI モデルが成熟し、産業化されており、NVIDIA GPU を用いてトレーニングされ、継続的に改善されている。成熟した AI モデルの例には、コンピュータービジョンモデル、音声認識、レコメンダーシステム、グラフとツリー、時系列モデル、生成モデル、可変エンコーダー、大規模言語モデルがある。実際、新しい言語や分野に合わせた大規模言語モデルのカスタマイズは、史上最大級のスーパーコンピューティング用途の一つになる可能性が高い。

NVIDIA の新しい [Omniverse™ プラットフォーム](https://developer.nvidia.com/nvidia-omniverse-platform)は、多数のメタバース環境を支え、膨大な GPU 計算能力を必要とする。Omniverse 対応の多くのメタバースでリアルタイムレンダリングとシミュレーションを担う NVIDIA RTX GPU に加えて、H100 対応システムが複雑なデジタルツインの課題にさらなる AI およびシミュレーション性能をもたらすと期待している。最大規模のスーパーコンピューティングへの取り組みの一つが NVIDIA 自身の [Earth-2 スーパーコンピュータープロジェクト](https://blogs.nvidia.com/blog/2021/11/12/earth-2-supercomputer/)であり、膨大な量のデータを、Omniverse で物理シミュレーションを実行する地球のデジタルツインへ継続的にストリーミングし、世界各地の将来の気象パターンを予測する。

<span id="figure-01"></span>

![現代のクラウドコンピューティングにおける多様なワークロード](../../papers/nvidia-h100-architecture/figure-01.png)

**図 1。** 現代のクラウドデータセンターのワークロードには NVIDIA GPU アクセラレーションが必要

本ホワイトペーパーでは、次世代かつ最高性能のデータセンター GPU である新しい NVIDIA H100 Tensor Core GPU を紹介する。NVIDIA Hopper GPU アーキテクチャを基盤とする H100 は、クラウドデータセンター、サーバー、エッジシステム、ワークステーションにおける AI のトレーニングと推論、HPC、データ分析アプリケーションを高速化する。

H100、新しい H100 ベースの DGX、DGX SuperPOD、HGX システム、そして新しい H100 ベースの Converged Accelerator の概要に続き、H100 のハードウェアアーキテクチャ、効率の改善、新しいプログラミング機能を詳しく掘り下げる。

<span id="section-2"></span>

## 2 NVIDIA H100 Tensor Core GPU の概要

人工知能（AI）、ハイパフォーマンスコンピューティング（HPC）、データ分析の複雑さは指数関数的に増しており、科学者やエンジニアは最先端のコンピューティングプラットフォームを使用する必要がある。NVIDIA Hopper GPU アーキテクチャは、低レイテンシーで最高性能のコンピューティングを安全に提供し、データセンター規模の計算に必要な機能をフルスタックで統合する。

NVIDIA Hopper GPU アーキテクチャを搭載した NVIDIA® H100 Tensor Core GPU は、NVIDIA のデータセンタープラットフォームにアクセラレーテッドコンピューティング性能の次なる飛躍をもたらす。H100 は、小規模な企業ワークロードからエクサスケール HPC、1 兆パラメーターの AI モデルまで、多様なワークロードを安全に高速化する。

NVIDIA 向けにカスタマイズされた TSMC の 4N プロセスを用い、800 億個のトランジスタと数多くのアーキテクチャ上の進歩を盛り込んだ H100 は、これまでに製造された世界で最も先進的なチップである。

<span id="figure-02"></span>

![SXM5 モジュールに搭載された NVIDIA H100 GPU](../../papers/nvidia-h100-architecture/figure-02.png)

**図 2。** 新しい SXM5 モジュール上の NVIDIA H100 GPU

H100 は NVIDIA の第 9 世代データセンター GPU であり、前世代の NVIDIA A100 Tensor Core GPU と比べて、大規模 AI および HPC で桁違いの性能向上を実現するよう設計されている。H100 は、AI および HPC ワークロードのストロングスケーリングを改善するという A100 の主要な設計方針を引き継ぎつつ、アーキテクチャ効率を大幅に向上させている。

現在主流の AI および HPC モデルでは、InfiniBand インターコネクトを備えた H100 が A100 の最大 30 倍の性能を発揮する（[図 3](#figure-03)を参照）。

複数の GPU アクセラレーテッドノードにまたがるモデル並列性を必要とする、最大級かつ最難関の計算ワークロードを対象とした新しい NVLink Switch System インターコネクトにより、これらのワークロードはさらに世代を超えた性能飛躍を得て、場合によっては InfiniBand 接続の H100 に対して再び 3 倍の性能を達成する。

<span id="figure-03"></span>

![HPC、AI 推論、AI トレーニングの各ワークロードにおける H100 の性能](../../papers/nvidia-h100-architecture/figure-03.png)

**図 3。** H100 は次世代の AI と HPC のブレークスルーを実現する。すべての性能値は現時点の予測に基づく暫定値であり、出荷製品では変更される可能性がある。A100 クラスター：HDR IB ネットワーク。H100 クラスター：表示箇所では NVLink Switch System を備えた NDR IB ネットワーク。NVLink Switch System 技術は現在 H100 システムでは利用できないが、システムと提供時期は今後発表される予定である。GPU 数：Climate Modeling 1K、LQCD 1K、Genomics 8、3D-FFT 256、MT-NLG 32（バッチサイズ：1 秒では A100 が 4、H100 が 60、1.5 秒および 2 秒では A100 が 8、H100 が 64）、MRCNN 8（バッチ 32）、GPT-3 16B 512（バッチ 256）、DLRM 128（バッチ 64K）、GPT-3 16K（バッチ 512）、MoE 8K（バッチ 512、GPU あたり 1 エキスパート）。

GTC Spring 2022 では、新製品 NVIDIA Grace Hopper Superchip が発表された。Hopper H100 Tensor Core GPU は、テラバイト規模のアクセラレーテッドコンピューティング専用に構築され、大規模モデルの AI および HPC で 10 倍高い性能を実現する NVIDIA Grace Hopper Superchip CPU+GPU アーキテクチャを駆動する。

NVIDIA Grace CPU は Arm® アーキテクチャの柔軟性を活かし、アクセラレーテッドコンピューティング向けにゼロから設計された CPU とサーバーアーキテクチャを実現する。H100 と Grace は NVIDIA の超高速チップ間インターコネクトで接続され、PCIe Gen5 の 7 倍に当たる 900 GB/s の帯域幅を提供する。この革新的な設計は、現在最速のサーバーと比べて総帯域幅を最大 30 倍に高め、数テラバイトのデータを扱うアプリケーションで最大 10 倍の性能を実現する。

<span id="figure-04"></span>

![Grace Hopper Superchip](../../papers/nvidia-h100-architecture/figure-04.png)

**図 4。** Grace Hopper Superchip

<span id="section-2-1"></span>

### 2.1 NVIDIA H100 GPU の主な機能の概要

- 新しい Streaming Multiprocessor（SM）には、性能と効率に関する多くの改善が施されている。主な新機能は次のとおりである。
  - 新しい第 4 世代 Tensor Core は、SM あたりの高速化、SM 数の増加、H100 の高いクロックを含め、チップ全体で A100 より最大 6 倍高速である。SM 単位では、Tensor Core は同等のデータ型で A100 SM の 2 倍の MMA（Matrix Multiply-Accumulate）演算レートを提供し、前世代の 16 ビット浮動小数点オプションと比較して、新しい FP8 データ型を用いると A100 の 4 倍のレートを提供する。Sparsity 機能は、ディープラーニングネットワークの細粒度構造化スパース性を活用し、標準的な Tensor Core 演算の性能を 2 倍にする。
  - 新しい DPX 命令は、Dynamic Programming アルゴリズムを A100 GPU より最大 7 倍高速化する。例として、ゲノミクス処理向けの Smith-Waterman アルゴリズムと、動的な倉庫環境でロボット群の最適経路を求める Floyd-Warshall アルゴリズムがある。
  - SM あたりのクロック当たり性能が 2 倍になったことに加え、SM 数の増加と H100 の高いクロックにより、チップ全体の IEEE FP64 および FP32 処理レートは A100 より 3 倍高速である。
  - 新しい Thread Block Cluster 機能により、単一 SM 上の単一 Thread Block より大きな粒度で局所性をプログラムから制御できる。これは、プログラミング階層に新たなレベルを追加し、Thread、Thread Block、Thread Block Cluster、Grid を含むように CUDA プログラミングモデルを拡張するものである。Cluster により、複数の SM にまたがって同時実行される複数の Thread Block が同期し、協調してデータを取得・交換できる。
  - 新しい非同期実行機能には、グローバルメモリと共有メモリの間で大きなデータブロックを非常に効率よく転送できる、新しい Tensor Memory Accelerator（TMA）ユニットが含まれる。TMA は Cluster 内の Thread Block 間の非同期コピーにも対応する。また、アトミックなデータ移動と同期を行う新しい Asynchronous Transaction Barrier も備える。
- 新しい Transformer Engine は、Transformer モデルのトレーニングと推論を高速化するために専用設計された、ソフトウェアとカスタム Hopper Tensor Core 技術の組み合わせを用いる。Transformer Engine は FP8 と 16 ビット演算をインテリジェントに管理して動的に選択し、各層で FP8 と 16 ビットの間の再キャストとスケーリングを自動処理することで、前世代の A100 と比べて大規模言語モデルの AI トレーニングを最大 9 倍、AI 推論を最大 30 倍高速化する。
- HBM3 メモリサブシステムは、前世代と比べて帯域幅を約 2 倍に高める。H100 SXM5 GPU は HBM3 メモリを備えた世界初の GPU であり、クラス最高の 3 TB/sec のメモリ帯域幅を提供する。
- 50 MB の L2 キャッシュアーキテクチャは、モデルやデータセットの大部分をキャッシュして繰り返しアクセスできるようにし、HBM3 へのアクセスを減らす。
- 第 2 世代 Multi-Instance GPU（MIG）技術は、A100 と比べて GPU Instance あたり約 3 倍の計算能力と約 2 倍のメモリ帯域幅を提供する。MIG レベルの Trusted Execution Environment（TEE）を備えた Confidential Computing 機能も初めて提供される。最大 7 個の独立した GPU Instance をサポートし、それぞれに専用の NVDEC および NVJPG ユニットを備える。各 Instance は NVIDIA 開発者ツールと連携する独自のパフォーマンスモニター群も備えるようになった。
- 新しい Confidential Computing 対応は、ユーザーデータを保護し、ハードウェアおよびソフトウェア攻撃を防ぎ、仮想化環境と MIG 環境で VM を相互に隔離・保護する能力を高める。H100 は世界初のネイティブ Confidential Computing GPU を実現し、CPU との Trusted Execution Environment を PCIe のフルラインレートで拡張する。
- 第 4 世代 NVIDIA NVLink® は、all-reduce 演算の帯域幅を 3 倍、一般的な帯域幅を前世代 NVLink より 50% 増加させ、マルチ GPU I/O に PCIe Gen 5 の 7 倍に当たる合計 900 GB/sec の帯域幅を提供する。
- 第 3 世代 NVSwitch 技術には、サーバー、クラスター、データセンター環境で複数の GPU を接続するため、ノード内外に配置されるスイッチが含まれる。ノード内の各 NVSwitch は、第 4 世代 NVLink リンク用に 64 ポートを提供し、マルチ GPU 接続を高速化する。スイッチの総スループットは前世代の 7.2 Tbits/sec から 13.6 Tbits/sec に増加する。新しい第 3 世代 NVSwitch 技術は、マルチキャストと NVIDIA SHARP のネットワーク内リダクションを用いる集合演算のハードウェアアクセラレーションも提供する。
- 新しい NVLink Switch System インターコネクト技術と、第 3 世代 NVSwitch 技術に基づく新しい第 2 レベル NVLink Switch は、アドレス空間の隔離と保護を導入し、最大 32 ノードまたは 256 GPU を 2:1 にテーパーされた fat tree トポロジーで NVLink 接続できるようにする。接続されたノードは、57.6 TB/sec の全対全帯域幅を提供し、驚異的な 1 exaFLOP の FP8 スパース AI 演算能力を供給できる。
- PCIe Gen 5 は、Gen 4 PCIe の合計 64 GB/sec（各方向 32 GB/sec）に対して、合計 128 GB/sec（各方向 64 GB/sec）の帯域幅を提供する。PCIe Gen 5 により、H100 は最高性能の x86 CPU および SmartNIC／DPU（Data Processing Unit）と接続できる。

ストロングスケーリングの改善、レイテンシーとオーバーヘッドの削減、GPU プログラミング全般の簡素化を目的とする、そのほかの新機能も数多く盛り込まれている。

本ホワイトペーパーの[第 3 節](#section-3)では、新しい H100 ベースの DGX、HGX、Converged Accelerator、AI スーパーコンピューティングシステムについて説明する。

[第 4 節](#section-4)では、H100 GPU のアーキテクチャ機能、新しいプログラミング能力、性能向上について詳述する。

<span id="figure-05"></span>

![Hopper H100 の 6 つの新技術](../../papers/nvidia-h100-architecture/figure-05.png)

**図 5。** Hopper H100 の新技術

<span id="section-3"></span>

## 3 NVIDIA GPU アクセラレーテッドデータセンター

AI やデータ分析からハイパフォーマンスコンピューティング（HPC）まで、データセンターは最重要課題の一部を解決する鍵となる。ハードウェアとソフトウェアにまたがって統合されたエンドツーエンドの NVIDIA アクセラレーテッドコンピューティングプラットフォームは、あらゆる最新ワークロードで開発からデプロイまでを支える、堅牢で安全なインフラストラクチャの設計図を企業に提供する。

ディープラーニングのデータセットは大規模化・複雑化しており、対話型 AI、レコメンダーシステム、コンピュータービジョンなどのワークロードが各業界でますます普及している。ハードウェアとソフトウェアを含む NVIDIA データセンタープラットフォームは、AI トレーニングを大幅に高速化し、データサイエンスチームの生産性向上、大幅なコスト削減、ROI 達成までの時間短縮をもたらす。

データセンターで推論ワークロードを高速化するには、スケールアウトして利用可能な計算リソースを余すところなく活用できる、俊敏で弾力的なインフラストラクチャが必要である。Multi-Instance GPU（MIG）などの新技術により、NVIDIA のソリューションは画像認識、レコメンダーシステム、自然言語処理などの推論ワークロードを高速化するうえで独自の優位性を持ち、AI をアプリケーションに導入するために必要な最高のスループットとリアルタイム応答性を提供する。

HPC は、データセンターにおける科学の進歩を支える最も重要なツールの一つである。NVIDIA GPU は現代の HPC データセンターを動かすエンジンである。より少ないサーバーで飛躍的な性能を実現し、知見獲得を迅速化してコストを劇的に削減することで、NVIDIA データセンタープラットフォームは科学的発見への道を切り開く。

企業はかつてない量のデータを生成・収集している。分析できるデータが多いほど、より多くを学べる。NVIDIA データセンタープラットフォームと分析ソリューションにより、企業はこれまで以上に速くデータから実用的な知見を引き出せる。

データセンター向け NVIDIA GPU アクセラレーションは、NVIDIA の広大なパートナーサーバーメーカーのエコシステムが提供する幅広いサーバーを通じて利用できる。H100 GPU は、サーバー設計ごとの異なる要件に対応するため、さまざまな構成で提供される。

以降の各節では、SMX5 および PCIe Gen 5 フォームファクターの H100 GPU、DGX H100 と DGX SuperPOD システム、HGX H100、NVIDIA H100 GPU の能力と NVIDIA® ConnectX-7 SmartNIC の高度なネットワーク機能を組み合わせた H100 CNX Converged Accelerator など、NVIDIA データセンター対応の H100 ベースのシステムとボードを簡潔に説明する。DGX H100 システムの詳細については、[第 7 節](#section-7)を参照されたい。

<span id="section-3-1"></span>

### 3.1 H100 SXM5 GPU

H100 GPU と HBM3 メモリスタックを搭載し、第 4 世代 NVLink と PCIe Gen 5 接続も提供する NVIDIA のカスタム設計 SXM5 ボードを用いた H100 SXM5 構成は、最高のアプリケーション性能を提供する。この構成は、サーバー内およびサーバー間で複数 GPU に拡張するアプリケーションを持つ顧客に最適である。4 GPU 構成と 8 GPU 構成の HGX H100 サーバーボードを通じて提供される。4 GPU 構成は GPU 間にポイントツーポイント NVLink 接続を備え、サーバー内の CPU 対 GPU 比を高める一方、8 GPU 構成は NVSwitch を備え、SHARP のネットワーク内リダクションと、任意の GPU ペア間で 900 GB/s のフル NVLink 帯域幅を提供する。H100 SXM5 GPU は、強力な新しい DGX H100 サーバーと DGX SuperPOD システムにも使用される。

<span id="section-3-2"></span>

### 3.2 H100 PCIe Gen 5 GPU

H100 PCIe Gen 5 構成は、わずか 350 W の熱設計電力（TDP）で H100 SXM5 GPU の全機能を提供する。この構成では、オプションで NVLink ブリッジを使用し、PCIe Gen5 のほぼ 5 倍となる 600 GB/s の帯域幅で最大 2 基の GPU を接続できる。サーバーあたりの消費電力を抑えて標準ラックに収容する主流のアクセラレーテッドサーバーに適しており、H100 PCIe は AI 推論や一部の HPC アプリケーションを含む、同時に 1 基または 2 基の GPU へ拡張するアプリケーションで優れた性能を発揮する。主要なデータ分析、AI、HPC アプリケーション 10 種の組み合わせでは、単一の H100 PCIe GPU が、消費電力を 50% に抑えながら H100 SXM5 GPU の実効性能の 65% を効率よく提供する。

<span id="section-3-3"></span>

### 3.3 DGX H100 と DGX SuperPOD

NVIDIA DGX H100 は、トレーニング、推論、分析に対応する汎用高性能 AI システムである。DGX H100 は Bluefield-3、NDR InfiniBand、第 2 世代 MIG 技術を搭載する。単一の DGX H100 システムは、比類のない 16 petaFLOPS の FP16 スパース AI 演算性能を発揮する。複数の DGX H100 システムを DGX POD、さらに DGX SuperPOD と呼ばれるクラスターへ接続することで、この性能を容易にスケールアップできる。DGX SuperPOD は「scalable unit」と呼ばれる 32 台の DGX H100 システムから始まり、第 3 世代 NVSwitch 技術に基づく新しい第 2 レベル NVLink Switch で接続された 256 基の H100 GPU を統合し、前例のない 1 exaFLOP の FP8 スパース AI 演算性能を提供する。DGX H100 SuperPOD は InfiniBand と NVLINK Switch の両ネットワークオプションに対応する。

詳細については、[第 7 節](#section-7)を参照されたい。

<span id="section-3-4"></span>

### 3.4 HGX H100

ワークロードの複雑さが爆発的に増すにつれ、複数の GPU が相互に極めて高速に通信しながら連携する必要がある。NVIDIA HGX H100™ は、複数の H100 GPU と NVLink および NVSwitch を利用した高速インターコネクトを組み合わせ、世界で最も強力なスケールアップサーバーの構築を可能にする。

HGX H100 は、4 基または 8 基の H100 GPU 構成を備えた統合ベースボードとして、サーバーのビルディングブロック向けに提供される。4 GPU の HGX H100 は、GPU 間を完全に相互接続するポイントツーポイント NVLink 接続を備え、8 GPU 構成は NVSwitch を介して GPU 間のフル帯域幅を提供する。H100 のマルチプレシジョン Tensor Core の能力を活用し、8-way HGX H100 はスパース FP8 演算で 32 petaFLOPS を超えるディープラーニング演算性能を提供する。HGX H100 は、さまざまなアプリケーションワークロードで予測可能な性能を発揮する標準化された高性能サーバーを可能にするとともに、NVIDIA のパートナーサーバーメーカーのエコシステムにおける市場投入期間の短縮も実現する。

<span id="section-3-5"></span>

### 3.5 H100 CNX Converged Accelerator

NVIDIA H100 CNX は、NVIDIA H100 GPU の能力と、最大 400 Gb/s の帯域幅を提供し、NVIDIA ASAP2（Accelerated Switching and Packet Processing）や TLS／IPsec／MACsec の暗号化・復号をインラインでハードウェアアクセラレーションする革新的な機能を備えた NVIDIA® ConnectX-7 SmartNIC の高度なネットワーク機能を組み合わせる。この独自のアーキテクチャは、企業データセンターの分散 AI トレーニングやエッジでの 5G 信号処理など、GPU を利用する I/O 集約型ワークロードに前例のない性能をもたらす。

<span id="section-4"></span>

## 4 NVIDIA H100 GPU アーキテクチャの詳細

新しい Hopper GPU アーキテクチャに基づく NVIDIA H100 GPU は、複数の革新を備える。

- 新しい第 4 世代 Tensor Core は、これまで以上に幅広い AI および HPC タスクで、かつてない速さの行列演算を実行する。
- 新しい Transformer Engine により、H100 は前世代の A100 と比べて、大規模言語モデルの AI トレーニングを最大 9 倍、AI 推論を最大 30 倍高速化する。
- 新しい NVLink Network インターコネクトは、複数の計算ノードにまたがる最大 256 基の GPU 間通信を可能にする。
- Secure MIG は GPU を隔離された適正規模のインスタンスに分割し、小規模ワークロードの QoS（Quality of Service）を最大化する。

NVIDIA の H100 は、真に非同期な初の GPU である。H100 は A100 のグローバルメモリから共有メモリへの非同期転送をすべてのアドレス空間へ拡張し、テンソルメモリアクセスパターンへの対応を追加する。これにより、データをチップへ入出力するエンドツーエンドの非同期パイプラインをアプリケーションが構築でき、データ移動を計算と完全にオーバーラップさせて隠蔽できる。

新しい Tensor Memory Accelerator を用いれば、H100 の全メモリ帯域幅を管理するのに必要な CUDA スレッドはごく少数となり、ほかの CUDA スレッドの大部分は、新世代 Tensor Core 向けデータの前処理や後処理など、汎用計算に集中できる。

H100 は、CUDA スレッドグループ階層に Thread Block Cluster と呼ばれる新しいレベルを追加する。Cluster は同時にスケジュールされることが保証された Thread Block のグループであり、複数の SM にまたがるスレッドの効率的な協調とデータ共有を可能にする。Cluster は Tensor Memory Accelerator や Tensor Core などの非同期ユニットも、より効率的に協調駆動する。

増え続けるオンチップアクセラレーターと多様な汎用スレッド群を連携させるには、同期が必要である。たとえば、出力を消費するスレッドとアクセラレーターは、それを生成するスレッドとアクセラレーターを待たなければならない。

NVIDIA の Asynchronous Transaction Barrier により、たとえ別々の SM に存在していても、Cluster 内の汎用 CUDA スレッドとオンチップアクセラレーターを効率よく同期できる。これらすべての新機能により、あらゆるユーザーとアプリケーションが H100 GPU の全ユニットを常時最大限に活用でき、H100 はこれまでで最も強力で、最もプログラムしやすく、電力効率に優れた GPU となる。

H100 GPU を駆動する完全版 GH100 GPU は、NVIDIA 向けにカスタマイズされた TSMC の 4N プロセスで製造され、800 億個のトランジスタ、814 mm² のダイサイズ、高周波数設計を備える。

NVIDIA GH100 GPU は、複数の GPU Processing Cluster（GPC）、Texture Processing Cluster（TPC）、Streaming Multiprocessor（SM）、L2 キャッシュ、HBM3 メモリコントローラーで構成される。

GH100 GPU の完全実装には、次のユニットが含まれる。

- 8 GPC、72 TPC（GPC あたり 9 TPC）、TPC あたり 2 SM、完全版 GPU あたり 144 SM
- SM あたり 128 FP32 CUDA Core、完全版 GPU あたり 18,432 FP32 CUDA Core
- SM あたり 4 基の第 4 世代 Tensor Core、完全版 GPU あたり 576 基
- 6 個の HBM3 または HBM2e スタック、12 基の 512 ビットメモリコントローラー
- 60 MB L2 キャッシュ
- 第 4 世代 NVLink および PCIe Gen 5

SXM5 ボードフォームファクターの NVIDIA H100 GPU には、次のユニットが含まれる。

- 8 GPC、66 TPC、TPC あたり 2 SM、GPU あたり 132 SM
- SM あたり 128 FP32 CUDA Core、GPU あたり 16,896 FP32 CUDA Core
- SM あたり 4 基の第 4 世代 Tensor Core、GPU あたり 528 基
- 80 GB HBM3、5 個の HBM3 スタック、10 基の 512 ビットメモリコントローラー
- 50 MB L2 キャッシュ
- 第 4 世代 NVLink および PCIe Gen 5

PCIe Gen 5 ボードフォームファクターの NVIDIA H100 GPU には、次のユニットが含まれる。

- 7 または 8 GPC、57 TPC、TPC あたり 2 SM、GPU あたり 114 SM
- SM あたり 128 FP32 CUDA Core、GPU あたり 14,592 FP32 CUDA Core
- SM あたり 4 基の第 4 世代 Tensor Core、GPU あたり 456 基
- 80 GB HBM2e、5 個の HBM2e スタック、10 基の 512 ビットメモリコントローラー
- 50 MB L2 キャッシュ
- 第 4 世代 NVLink および PCIe Gen 5

TSMC 4N 製造プロセスの採用により、H100 は GPU コア周波数を高め、ワット当たり性能を改善し、TSMC 7 nm N7 プロセスに基づく前世代 GA100 GPU より多くの GPC、TPC、SM を搭載できる。

[図 6](#figure-06)は、144 SM を備えた完全版 GH100 GPU を示す。H100 SXM5 GPU は 132 SM、PCIe 版は 114 SM を備える。H100 GPU は主に AI、HPC、データ分析向けのデータセンターおよびエッジ計算ワークロードの実行を目的に構築され、グラフィックス処理用ではないことに注意されたい。SXM5 と PCIe の両 H100 GPU でグラフィックス処理が可能なのは 2 TPC のみである（すなわち、頂点、ジオメトリ、ピクセルシェーダーを実行できる）。

<span id="figure-06"></span>

![144 SM を備えた GH100 完全版 GPU のブロック図](../../papers/nvidia-h100-architecture/figure-06.png)

**図 6。** 144 SM を備えた GH100 完全版 GPU

<span id="section-4-1"></span>

### 4.1 H100 SM アーキテクチャ

NVIDIA A100 Tensor Core GPU の SM アーキテクチャを基盤として、H100 SM は FP8 の導入により A100 の SM あたりピーク浮動小数点演算能力を 4 倍にし、従来のすべての Tensor Core および FP32／FP64 データ型では、同一クロックで A100 の SM 生演算能力を 2 倍にする。

新しい Transformer Engine と Hopper の FP8 Tensor Core の組み合わせにより、前世代の A100 と比べて、大規模言語モデルの AI トレーニングは最大 9 倍、AI 推論は 30 倍高速になる。Hopper の新しい DPX 命令は、ゲノミクスおよびタンパク質配列解析向け Smith-Waterman アルゴリズムの処理を最大 7 倍高速化する。

Hopper の新しい第 4 世代 Tensor Core、Tensor Memory Accelerator、そのほか多数の新しい SM および H100 全体のアーキテクチャ改善により、それ以外の多くの場合でも HPC および AI の性能が最大 3 倍向上する。

<span id="table-01"></span>

![NVIDIA H100 Tensor Core GPU の性能仕様](../../papers/nvidia-h100-architecture/table-01.png)

**表 1。** NVIDIA H100 Tensor Core GPU の性能仕様

<span id="figure-07"></span>

![GH100 Streaming Multiprocessor のブロック図](../../papers/nvidia-h100-architecture/figure-07.png)

**図 7。** GH100 Streaming Multiprocessor（SM）

<span id="section-4-1-1"></span>

#### 4.1.1 H100 SM の主な機能の概要

- 第 4 世代 Tensor Core：
  - SM あたりの高速化、SM 数の増加、H100 の高いクロックを含め、チップ全体で A100 より最大 6 倍高速。
  - SM 単位では、Tensor Core は同等のデータ型で A100 SM の 2 倍の MMA（Matrix Multiply-Accumulate）演算レートを提供し、前世代の 16 ビット浮動小数点オプションと比較して、新しい FP8 データ型を用いると A100 の 4 倍のレートを提供する。
  - Sparsity 機能は、ディープラーニングネットワークの細粒度構造化スパース性を活用し、標準的な Tensor Core 演算の性能を 2 倍にする。
- 新しい DPX 命令は、Dynamic Programming アルゴリズムを A100 GPU より最大 7 倍高速化する。例として、ゲノミクス処理向けの Smith-Waterman アルゴリズムと、動的な倉庫環境でロボット群の最適経路を求める Floyd-Warshall アルゴリズムがある。
- SM あたりのクロック当たり性能が 2 倍になったことに加え、SM 数の増加と H100 の高いクロックにより、チップ全体の IEEE FP64 および FP32 処理レートは A100 より 3 倍高速。
- 共有メモリと L1 データキャッシュの合計は 256 KB で、A100 より 1.33 倍大きい。
- 新しい非同期実行機能には、グローバルメモリと共有メモリの間で大きなデータブロックを効率よく転送できる、新しい Tensor Memory Accelerator（TMA）ユニットが含まれる。TMA は Cluster 内の Thread Block 間の非同期コピーにも対応する。また、アトミックなデータ移動と同期を行う新しい Asynchronous Transaction Barrier も備える。
- 新しい Thread Block Cluster 機能は、複数の SM にまたがる局所性の制御を公開する。
- Distributed Shared Memory により、複数の SM の共有メモリブロックにまたがるロード、ストア、アトミック演算で、SM 間を直接通信できる。

<span id="section-4-1-2"></span>

#### 4.1.2 H100 Tensor Core アーキテクチャ

Tensor Core は、行列の乗算・累積（MMA）演算に特化した高性能演算コアで、AI および HPC アプリケーションに画期的な性能をもたらす。1 基の NVIDIA GPU 内で複数の SM にわたり並列動作する Tensor Core は、標準の浮動小数点（FP）、整数（INT）、FMA（Fused Multiply-Accumulate）演算と比べて、スループットと効率を大幅に向上させる。Tensor Core は NVIDIA Tesla® V100 GPU で初めて導入され、新しい NVIDIA GPU アーキテクチャの世代ごとに強化されてきた。

H100 の新しい第 4 世代 Tensor Core アーキテクチャは、A100 と比べて、同一クロックで SM あたりの密行列およびスパース行列演算の生スループットを 2 倍にし、A100 より高い H100 の GPU Boost クロックを考慮すると、さらに大きな差となる。FP8、FP16、BF16、TF32、FP64、INT8 の MMA データ型をサポートする。新しい Tensor Core はデータ管理も効率化されており、オペランド供給電力を最大 30% 削減する。

<span id="figure-08"></span>

![A100 と H100 の FP16 Tensor Core 構造](../../papers/nvidia-h100-architecture/figure-08.png)

**図 8。** H100 FP16 Tensor Core は A100 FP16 Tensor Core と比べて 3 倍のスループットを実現

<span id="section-4-1-3"></span>

#### 4.1.3 Hopper FP8 データ形式

H100 GPU は、AI のトレーニングと推論の両方を高速化する FP8 Tensor Core を追加する。[図 9](#figure-09)に示すように、FP8 Tensor Core は FP32 および FP16 アキュムレーターと、2 種類の新しい FP8 入力型をサポートする。

- 指数部 4 ビット、仮数部 3 ビット、符号 1 ビットの E4M3
- 指数部 5 ビット、仮数部 2 ビット、符号 1 ビットの E5M2。

E4M3 はより高い精度と少ないダイナミックレンジを必要とする計算をサポートし、E5M2 はより広いダイナミックレンジと低い精度を提供する。FP8 は FP16 または BF16 と比べてデータ保存要件を半減し、スループットを 2 倍にする。

新しい Transformer Engine（後の節で説明）は FP8 と FP16 の両精度を利用し、大規模言語モデルなどの精度を維持しながら、メモリ使用量を削減して性能を向上させる。

<span id="figure-09"></span>

![Hopper FP8 精度形式とアキュムレーター型](../../papers/nvidia-h100-architecture/figure-09.png)

**図 9。** 新しい Hopper FP8 精度—H100 FP16／BF16 の 2 倍のスループットと半分のフットプリント

<span id="figure-10"></span>

![A100 FP16 と H100 FP8 の Tensor Core 構造](../../papers/nvidia-h100-architecture/figure-10.png)

**図 10。** H100 FP8 Tensor Core は A100 FP16 Tensor Core と比べて 6 倍のスループットを実現

<span id="figure-11"></span>

![H100 TF32、FP64、INT8 Tensor Core のスループット比較](../../papers/nvidia-h100-architecture/figure-11.png)

**図 11。** H100 の TF32、FP64、INT8 Tensor Core はすべて A100 の 3 倍のスループットを実現

複数のデータ型における A100 に対する H100 の演算高速化を、以下の[表 2](#table-02)に示す。

<span id="table-02"></span>

![複数のデータ型における A100 に対する H100 の高速化](../../papers/nvidia-h100-architecture/table-02.png)

**表 2。** A100 に対する H100 の高速化（H100 Performance、TC＝Tensor Core）

<span id="section-4-1-4"></span>

#### 4.1.4 Dynamic Programming を高速化する新しい DPX 命令

多くの「総当たり」最適化アルゴリズムには、より大きな問題を解く際に部分問題の解を何度も再利用するという性質がある。Dynamic Programming は、複雑な再帰問題をより単純な部分問題へ分解して解くアルゴリズム技法である。部分問題の結果を保存し、後で必要になったときに再計算する必要をなくすことで、Dynamic Programming アルゴリズムは指数規模の問題集合の計算複雑性を線形規模にまで削減する。

Dynamic Programming は、幅広い最適化、データ処理、ゲノミクスのアルゴリズムで一般的に用いられる。急速に成長するゲノムシーケンシング分野では、Smith-Waterman 動的計画法アルゴリズムが最も重要な手法の一つである。ロボティクス分野では、Floyd-Warshall は動的な倉庫環境でロボット群の最適経路をリアルタイムに求めるための主要アルゴリズムである。

H100 は、Dynamic Programming アルゴリズムの性能を Ampere GPU と比べて最大 7 倍高速化する DPX 命令を導入する。これらの新命令は、多数の DP アルゴリズムの内部ループに向けた高度な融合オペランドをサポートする。これにより、疾病診断、物流経路最適化、さらにはグラフ分析で、解を得るまでの時間が劇的に短縮される。

<span id="figure-12"></span>

![DPX 命令の用途と高速化](../../papers/nvidia-h100-architecture/figure-12.png)

**図 12。** DPX 命令が Dynamic Programming を高速化

<span id="section-4-1-5"></span>

#### 4.1.5 統合 L1 データキャッシュと共有メモリ

Volta V100 で初めて導入された NVIDIA の統合 L1 データキャッシュおよび共有メモリサブシステムアーキテクチャは、性能を大幅に向上させると同時に、プログラミングを簡素化し、アプリケーション性能をピークまたはその近傍まで引き出すために必要なチューニングを減らす。データキャッシュと共有メモリの機能を単一のメモリブロックに統合することで、どちらの種類のメモリアクセスにも最良の総合性能を提供する。L1 データキャッシュと共有メモリの合計容量は、A100 の 192 KB/SM に対して H100 では 256 KB/SM である。SM 共有メモリ自体のサイズは、H100 では最大 228 KB に設定できる。

<span id="section-4-1-6"></span>

#### 4.1.6 H100 の計算性能概要

総合すると、H100 のあらゆる新しい計算技術の進歩を考慮した場合、H100 は A100 に対して約 6 倍の計算性能向上を実現する。[図 13](#figure-13)は、まず 132 SM によって A100 の 108 SM より SM 数が 22% 増加し、そこから積み上がる H100 の改善を要約している。新しい第 4 世代 Tensor Core により、H100 の各 SM は 2 倍高速である。さらに各 Tensor Core 内では、新しい FP8 形式と対応する Transformer Engine がもう一段 2 倍の改善をもたらす。最後に、H100 のクロック周波数向上がさらに約 1.3 倍の性能向上を実現する。これらの改善を合わせると、H100 は A100 の約 6 倍のピーク計算スループットを達成し、世界で最も計算負荷の高いワークロードに大きな飛躍をもたらす。

<span id="figure-13"></span>

![A100 に対して積み上がる H100 の計算性能向上](../../papers/nvidia-h100-architecture/figure-13.png)

**図 13。** H100 の計算性能向上の概要。H100 は世界で最も計算負荷の高いワークロードに 6 倍のスループットを提供する。

<span id="section-4-2"></span>

### 4.2 H100 GPU の階層と非同期性の改善

並列プログラムで高性能を達成するために不可欠な二つの鍵は、データ局所性と非同期実行である。プログラムデータを実行ユニットのできるだけ近くへ移動することで、プログラマーはローカルデータへの低レイテンシーかつ高帯域幅のアクセスがもたらす性能を活かせる。非同期実行では、メモリ転送やほかの処理とオーバーラップできる独立したタスクを見つける。目標は、GPU 内のすべてのユニットを完全に稼働させ続けることである。ここでは Hopper の GPU プログラミング階層に追加され、単一 SM 上の単一 Thread Block より大きな規模の局所性を公開する重要な新しい階層を取り上げる。また、性能を向上させ、同期のオーバーヘッドを削減する新しい非同期実行機能についても説明する。

<span id="section-4-2-1"></span>

#### 4.2.1 Thread Block Cluster

CUDA プログラミングモデルは長年にわたり、プログラム内の局所性を活かすため、複数の Thread Block を含む Grid を使用する GPU 計算アーキテクチャに依存してきた。Thread Block は単一の SM 上で同時実行される複数のスレッドを含み、スレッドは高速なバリアで同期し、SM の共有メモリを用いてデータを交換できる。しかし、GPU が 100 SM を超える規模へ成長し、計算プログラムが複雑化するにつれ、プログラミングモデルで表現される局所性の単位が Thread Block だけでは、実行効率を最大化するには不十分である。

H100 は、単一 SM 上の単一 Thread Block より大きな粒度で局所性を制御できる、新しい Thread Block Cluster アーキテクチャを導入する。Thread Block Cluster は CUDA プログラミングモデルを拡張し、GPU の物理プログラミング階層に新たなレベルを加え、Thread、Thread Block、Thread Block Cluster、Grid を含むようにする。Cluster は SM のグループへ同時にスケジュールされることが保証された Thread Block のグループであり、複数の SM にまたがるスレッドの効率的な協調を可能にすることを目的とする。

H100 の Cluster は GPC 内の複数 SM にまたがって同時実行される。GPC はハードウェア階層において常に物理的に近接する SM のグループである。Cluster には、ハードウェアアクセラレーションされたバリアと、以降の節で説明する新しいメモリアクセス協調機能がある。GPC 内の SM 専用の SM 間ネットワークにより、Cluster 内のスレッド間で高速にデータを共有できる。CUDA では、[図 14](#figure-14)に示すように、Grid 内の Thread Block をカーネル起動時に任意で Cluster へグループ化でき、Cluster の機能は CUDA の [cooperative_groups API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups)から利用できる。

<span id="figure-14"></span>

![従来の Grid と Thread Block Cluster を備えた H100 の Grid](../../papers/nvidia-h100-architecture/figure-14.png)

**図 14。** Thread Block Cluster と Cluster を備えた Grid。上図の左半分に示す A100 のような従来の CUDA プログラミングモデルでは、Grid は Thread Block で構成される。Hopper アーキテクチャは、図の右半分に示す任意選択の Cluster 階層を追加する。

<span id="section-4-2-2"></span>

#### 4.2.2 Distributed Shared Memory

Cluster では、すべてのスレッドがロード、ストア、アトミック演算を用いて、ほかの SM の共有メモリへ直接アクセスできる。この機能は Distributed Shared Memory（DSMEM）と呼ばれる。共有メモリの仮想アドレス空間が Cluster 内のすべての Block に論理的に分散しているためである。DSMEM により SM 間のデータ交換が効率化され、データを受け渡すためにグローバルメモリへ書き込み、そこから読み出す必要がなくなる。Cluster 専用の SM 間ネットワークは、リモート DSMEM への高速かつ低レイテンシーなアクセスを保証する。グローバルメモリを使用する場合と比べて、DSMEM は Thread Block 間のデータ交換を約 7 倍高速化する。

<span id="figure-15"></span>

![A100 と H100 における Thread Block 間のデータ交換](../../papers/nvidia-h100-architecture/figure-15.png)

**図 15。** Thread Block 間のデータ交換（A100 対 Cluster を備えた H100）

CUDA レベルでは、Cluster 内のすべての Thread Block の全 DSMEM セグメントが各スレッドの汎用アドレス空間へマッピングされるため、DSMEM 全体を単純なポインターで直接参照できる。CUDA ユーザーは cooperative_groups API を利用し、Cluster 内の任意の Thread Block への汎用ポインターを構築できる。DSMEM 転送は、完了追跡用の共有メモリベースのバリアと同期する非同期コピー操作として表現することもできる。

以下の[図 16](#figure-16)は、さまざまなアルゴリズムで Cluster を使用する性能上の利点を示す。Cluster は、プログラマーが単一の SM だけでなく GPU のより大きな部分を直接制御できるようにして、性能を向上させる。Cluster では、単一の Thread Block だけでは不可能な、より多数のスレッドとより大きな共有メモリプールへのアクセスを伴う協調実行が可能になる。

<span id="figure-16"></span>

![Cluster 使用時と非使用時の性能](../../papers/nvidia-h100-architecture/figure-16.png)

**図 16。** Cluster 使用時と非使用時の性能比較。H100 の暫定性能推定値は現時点の予測に基づいており、出荷製品では変更される可能性がある。

<span id="section-4-2-3"></span>

#### 4.2.3 非同期実行

NVIDIA GPU の新しい世代にはそれぞれ、性能、プログラマビリティ、電力効率、GPU 利用率、そのほか多くの要素を改善する多数のアーキテクチャ強化が含まれる。近年の NVIDIA GPU 世代には、データ移動、計算、同期をさらにオーバーラップできる非同期実行機能が含まれてきた。Hopper アーキテクチャは、非同期実行を改善し、メモリコピーを計算やほかの独立した処理とさらにオーバーラップさせると同時に、同期ポイントを最小限に抑える新機能を提供する。

以下では、Tensor Memory Accelerator（TMA）と呼ばれる新しい非同期メモリコピーユニットと、新しい Async Transaction Barrier について説明する。

<span id="figure-17"></span>

![Hopper における非同期実行の並行性](../../papers/nvidia-h100-architecture/figure-17.png)

**図 17。** Hopper における非同期実行の並行性と強化。データ移動、計算、同期をプログラムからオーバーラップさせる。非同期の並行性と同期ポイントの最小化が性能の鍵となる。

<span id="section-4-2-4"></span>

#### 4.2.4 Tensor Memory Accelerator（TMA）

強力な新しい H100 Tensor Core へデータを供給しやすくするため、新しい Tensor Memory Accelerator（TMA）によってデータ取得効率が改善される。TMA は、大きなデータブロックと多次元テンソルをグローバルメモリから共有メモリへ、またその逆方向へ転送できる。

TMA 操作は、要素ごとのアドレス指定ではなく、テンソル次元とブロック座標を用いてデータ転送を指定するコピー記述子によって起動される（以下の[図 18](#figure-18)を参照）。大きなデータブロック（共有メモリ容量まで）を指定し、グローバルメモリから共有メモリへロードしたり、共有メモリからグローバルメモリへ保存したりできる。TMA はアドレス指定のオーバーヘッドを大幅に削減し、さまざまなテンソルレイアウト（1D-5D テンソル）、メモリアクセスモード、リダクション、そのほかの機能をサポートすることで効率を向上させる。

<span id="figure-18"></span>

![コピー記述子を介した TMA のアドレス生成](../../papers/nvidia-h100-architecture/figure-18.png)

**図 18。** コピー記述子による TMA アドレス生成

TMA 操作は非同期であり、A100 で導入された共有メモリベースの非同期バリアを利用する。さらに、TMA プログラミングモデルは単一スレッド方式で、warp 内から選出された一つのスレッドが、テンソルをコピーする非同期 TMA 操作（[`cuda::memcpy_async`](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#async_data_operations)）を発行し、その後に複数のスレッドが [`cuda::barrier`](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#aw-barrier) でデータ転送の完了を待てる。性能をさらに向上させるため、H100 SM はこれらの非同期バリア待機操作を高速化するハードウェアを追加している。

TMA の主な利点は、スレッドがほかの独立した処理を実行できるようになることである。[図 19](#figure-19)の左側に示す A100 では、非同期メモリコピーは特別な LoadGlobalStoreShared 命令を用いて実行されるため、スレッドがすべてのアドレスを生成し、コピー領域全体をループする役割を担っていた。

Hopper では、TMA がすべてを処理する。一つのスレッドが TMA を起動する前にコピー記述子を作成し、それ以降のアドレス生成とデータ移動はハードウェアで処理される。TMA は、テンソルのセグメントをコピーする際のストライド、オフセット、境界の計算を引き受けるため、はるかに単純なプログラミングモデルを提供する。

<span id="figure-19"></span>

![H100 の TMA と A100 の LDGSTS による非同期メモリコピー](../../papers/nvidia-h100-architecture/figure-19.png)

**図 19。** H100 の TMA と A100 の LDGSTS による非同期メモリコピーの比較

<span id="section-4-2-5"></span>

#### 4.2.5 Asynchronous Transaction Barrier

Asynchronous Barrier は当初 Ampere GPU アーキテクチャで導入された。[図 20](#figure-20)の左側を参照されたい。一群のスレッドがデータを生成し、バリアの後ですべてのスレッドがそのデータを消費する例を考える。Asynchronous Barrier は同期プロセスを二つの手順に分ける。まず、各スレッドは共有データの自分の担当部分を生成し終えると「Arrive」を通知する。この「Arrive」は非ブロッキングであるため、スレッドはほかの独立した処理を自由に実行できる。

最終的に、スレッドはほかのすべてのスレッドが生成したデータを必要とする。この時点で「Wait」を実行し、すべてのスレッドが「Arrive」を通知するまでブロックされる。

Asynchronous Barrier の利点は、早く到着したスレッドが待機中に独立した処理を実行できることである。このオーバーラップが追加性能の源となる。すべてのスレッドに十分な独立処理があれば、すでに全スレッドが Arrive 済みなので Wait 命令を即座にリタイアでき、バリアは事実上「無料」になる。

Hopper で新たに加わったのは、ほかのすべてのスレッドが到着するまで「Waiting」中のスレッドをスリープさせる機能である。従来のチップでは、Waiting スレッドは共有メモリ内のバリアオブジェクトをスピンしていた。

Asynchronous Barrier は引き続き Hopper プログラミングモデルの一部だが、Hopper は Asynchronous Transaction Barrier と呼ばれる新しい形式のバリアを追加する。Asynchronous Transaction Barrier は Asynchronous Barrier と非常によく似ている。[図 20](#figure-20)の右側を参照されたい。これも分割バリアだが、スレッドの到着だけでなくトランザクションもカウントする。Hopper には、書き込むデータとトランザクション数の両方を渡す共有メモリ書き込み用の新しいコマンドが含まれる。トランザクション数は本質的にはバイト数である。Asynchronous Transaction Barrier は、すべての生成側スレッドが Arrive を実行し、全トランザクション数の合計が期待値に達するまで、Wait コマンドでスレッドをブロックする。

Asynchronous Transaction Barrier は、非同期メモリコピーやデータ交換のための強力な新しいプリミティブである。前述のように、Cluster は暗黙的な同期を伴うデータ交換のために Thread Block 間通信を実行でき、その Cluster 機能は Asynchronous Transaction Barrier の上に構築されている。

<span id="figure-20"></span>

![A100 Asynchronous Barrier と H100 Asynchronous Transaction Barrier](../../papers/nvidia-h100-architecture/figure-20.png)

**図 20。** A100 の Asynchronous Barrier と H100 の Asynchronous Transaction Barrier の比較

<span id="section-4-3"></span>

### 4.3 H100 の HBM および L2 キャッシュメモリアーキテクチャ

GPU のメモリアーキテクチャと階層の設計はアプリケーション性能にとって重要であり、GPU のサイズ、コスト、消費電力、プログラマビリティに影響する。GPU には、オフチップ DRAM（フレームバッファ）の大容量デバイスメモリから、さまざまなレベルと種類のオンチップメモリ、SM の計算で用いるレジスタファイルまで、多様なメモリサブシステムが存在する。

高性能な HBM3 と HBM2e は、それぞれ H100 SXM5 GPU と PCIe H100 GPU で用いられる DRAM 技術である。HBM メモリは GPU と同じ物理パッケージに配置されたメモリスタックで構成され、従来の GDDR5/6 メモリ設計と比べて電力と面積を大幅に節約し、システムへより多くの GPU を搭載できるようにする。

CUDA プログラムがアクセスするグローバルメモリ領域とローカルメモリ領域は HBM メモリ空間に存在し、CUDA の用語では「デバイスメモリ」と呼ばれる。コンスタントメモリ空間はデバイスメモリに存在し、コンスタントキャッシュにキャッシュされる。テクスチャメモリ空間とサーフェスメモリ空間はデバイスメモリに存在し、テクスチャキャッシュにキャッシュされる。Level 2（L2）キャッシュは HBM（デバイス）メモリへの読み書きをキャッシュし、GPU 内のさまざまなサブシステムからのメモリ要求を処理する。HBM と L2 のメモリ空間には、すべての SM と GPU 上で実行されるすべてのアプリケーションからアクセスできる。

<span id="section-4-3-1"></span>

#### 4.3.1 H100 HBM3 および HBM2e DRAM サブシステム

HPC、AI、データ分析のデータセットが大規模化し続け、計算問題がますます複雑になるにつれ、より大きな GPU メモリ容量と帯域幅が不可欠になる。NVIDIA P100 は高帯域幅 HBM2 メモリ技術をサポートした世界初の GPU アーキテクチャであり、NVIDIA V100 はさらに高速で効率的かつ大容量な HBM2 実装を提供した。NVIDIA A100 GPU は HBM2 の性能と容量をさらに向上させた。

H100 SXM5 GPU は、高速な HBM3 メモリ 80 GB（5 スタック）をサポートし、3 TB/sec を超えるメモリ帯域幅を実現することで基準を大幅に引き上げ、わずか 2 年前に発売された A100 のメモリ帯域幅を実質的に 2 倍にする。PCIe H100 は、2 TB/sec を超えるメモリ帯域幅を備えた高速な HBM2e を 80 GB 提供する。

<span id="figure-21"></span>

![GPU 世代別の実効 HBM 帯域幅](../../papers/nvidia-h100-architecture/figure-21.png)

**図 21。** 世界初の HBM3 GPU メモリアーキテクチャ、実効帯域幅は 2 倍。メモリデータレートは未確定であり、最終製品では変更される可能性がある。

<span id="section-4-3-2"></span>

#### 4.3.2 H100 L2 キャッシュ

H100 の 50 MB L2 キャッシュは、A100 の 40 MB L2 より 1.25 倍大きい。モデルやデータセットのさらに大きな部分をキャッシュして繰り返しアクセスできるようにし、HBM3 または HBM2e DRAM へのアクセスを減らして性能を向上させる。分割クロスバー構造を用いることで、L2 キャッシュは、そのパーティションへ直接接続された GPC 内の SM からのメモリアクセスに対してデータを局所化し、キャッシュする。L2 キャッシュ常駐制御は容量利用率を最適化し、キャッシュに残すべきデータと退避すべきデータをプログラマーが選択的に管理できるようにする。

HBM3 または HBM2e DRAM と L2 キャッシュの両サブシステムは、メモリおよびキャッシュの使用効率と性能を最適化するデータ圧縮・展開技術をサポートする。

<span id="section-4-3-3"></span>

#### 4.3.3 メモリサブシステムの RAS 機能

H100 の HBM3 および HBM2e メモリサブシステムには、次の二つの主要な RAS（Reliability、Availability、Serviceability）機能が実装されている。

**ECC メモリの耐障害性。** H100 の HBM3/2e メモリサブシステムは、データを保護するため、1 ビット誤り訂正・2 ビット誤り検出（SECDED）誤り訂正符号（ECC）をサポートする。ECC は、データ破損の影響を受けやすい計算アプリケーションに高い信頼性をもたらす。GPU が非常に大きなデータセットを処理したり、アプリケーションを長時間実行したりする大規模クラスターコンピューティング環境では、特に重要である。H100 は HBM3/2e メモリで「Sideband ECC」をサポートする。これは、メイン HBM メモリから分離された小さなメモリ領域を ECC ビットに使用する方式である（メインメモリの一部を切り出して ECC ビットを格納する「Inline ECC」とは対照的である）。H100 のほかの主要メモリ構造も SECDED ECC で保護され、L2 キャッシュ、およびすべての SM 内の L1 キャッシュとレジスタファイルが含まれる。

**メモリ行の再マッピング。** H100 の HBM3/HBM2e サブシステムは、ECC エラーを生成したメモリセルを含むメモリ行を無効化し、起動時に行再マッピングロジックを用いて予約済みの正常な行へ置き換えられる。各 HBM3/HBM2e メモリバンクでは複数のメモリ行が予備行として確保され、不良と判定された行を置き換える必要がある場合に有効化できる。

<span id="table-03"></span>

![NVIDIA A100 と H100 データセンター GPU の比較](../../papers/nvidia-h100-architecture/table-03.png)

**表 3。** NVIDIA A100 と H100¹ データセンター GPU の比較

> **注：** H100 および A100 Tensor Core GPU は、AI および HPC の計算ワークロードを処理する高性能サーバやデータセンターラックに搭載するよう設計されているため、ディスプレイコネクタ、レイトレーシングアクセラレーション用の NVIDIA RT Core、NVENC エンコーダを備えていない。

<span id="section-4-4"></span>

### 4.4 Compute Capability

H100 GPU は新しい Compute Capability 9.0 をサポートする。[表 4](#table-04)では、NVIDIA GPU アーキテクチャの各 Compute Capability のパラメーターを比較する。

<span id="table-04"></span>

![V100、A100、H100 の Compute Capability 比較](../../papers/nvidia-h100-architecture/table-04.png)

**表 4。** Compute Capability：V100 対 A100 対 H100

<span id="section-4-5"></span>

### 4.5 第 2 世代 Secure MIG

NVIDIA Multi-Instance GPU（MIG）技術は、NVIDIA Ampere アーキテクチャを基盤とする A100 Tensor Core GPU で導入された。MIG は、同一 GPU を共有する複数のユーザーに独立し、完全に隔離された安全な GPU Instance を提供することで、Cloud Service Provider（CSP）のデータセンターをスケールアウトするための極めて重要な機能となっている。

<span id="section-4-5-1"></span>

#### 4.5.1 MIG 技術の振り返り

MIG 技術では、A100 または H100 の各 GPU（H100 SXM5 版と H100 PCIe 版の両方）を最大 7 個の GPU Instance に分割し、GPU 利用率を最適化できる。また、異なるクライアント（VM、コンテナ、プロセスなど）の間に規定された QoS と隔離を提供する。MIG はマルチテナントのユースケースを持つ Cloud Service Provider に特に有用であり、セキュリティを強化して顧客への GPU 利用率保証を可能にするだけでなく、あるクライアントがほかのクライアントの処理やスケジューリングへ影響を及ぼさないようにする。

<span id="figure-22"></span>

![CSP における Multi-Instance GPU 構成の例](../../papers/nvidia-h100-architecture/figure-22.png)

**図 22。** CSP MIG 構成の例。この CSP MIG 図は、同一または異なる組織の複数の独立ユーザーへ、単一の物理 GPU 内で各自専用の保護・隔離された GPU Instance を割り当てる方法を示す。

vGPU（仮想 GPU）の仮想マシン（VM）構成を管理、調整、保守、負荷分散するうえで重要な MIG 機能は、単一 GPU 上の GPU Instance 間、さらに一般的にはクラスター内の異なる GPU 間で vGPU を移行する機能である。

各 GPU Instance には、メモリシステム全体を通る独立して隔離された経路がある。オンチップクロスバーのポート、L2 キャッシュバンク、メモリコントローラー、DRAM アドレスバスは、いずれも個々のインスタンスに一意に割り当てられる。このため、ほかのタスクが自身のキャッシュを激しく入れ替えたり、DRAM インターフェースを飽和させたりしていても、個々のユーザーのワークロードは同じ L2 キャッシュ割り当てと DRAM 帯域幅を維持し、予測可能なスループットとレイテンシーで実行できる。

（基本的な MIG 技術の詳細については、[NVIDIA A100 Tensor Core GPU ホワイトペーパー](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)を参照されたい。）

<span id="section-4-5-2"></span>

#### 4.5.2 H100 MIG の強化

H100 の新しい第 2 世代 MIG 技術は、A100 と比べて GPU Instance あたり約 3 倍の計算能力と約 2 倍のメモリ帯域幅を提供する。NVIDIA Hopper アーキテクチャは MIG 技術を強化し、完全に安全でクラウドネイティブなマルチテナント・マルチユーザー MIG 構成を提供する。ハードウェアおよびハイパーバイザーレベルの新しい Confidential Computing 機能により、最大 7 個の GPU Instance を互いに安全に隔離できる（Confidential Computing の詳細については、後述の[第 5 節](#section-5)を参照）。

[図 23](#figure-23)は、単一 GPU を共有する複数のユーザーに対し、CPU と GPU が協調して複数の Trusted Execution Environment（TEE）を提供するシステム構成の例を示す。CPU 側は安全な NVIDIA ドライバーを備えた複数の Confidential VM を提供する。この例では、H100 GPU が四つの Secure MIG インスタンスに分割されている。CPU と GPU の間の転送は暗号化される。GPU ハードウェア仮想化は PCIe SR-IOV（MIG Instance あたり一つの Virtual Function（VF））を用いて提供される。機密性とデータ整合性は複数のハードウェアベースのセキュリティ機能によって提供され、ハードウェアファイアウォールが GPU Instance 間のメモリを隔離する。

<span id="figure-23"></span>

![単一の H100 GPU 上に 4 テナントを配置した Secure MIG の例](../../papers/nvidia-h100-architecture/figure-23.png)

**図 23。** マルチテナントの単一 GPU 構成における Secure MIG の例

Hopper アーキテクチャでは、各 GPU Instance に専用の画像および動画デコーダーを割り当て、共有インフラストラクチャ上で安全かつ高スループットな Intelligent Video Analytics（IVA）を実現できるようになった。各 MIG GPU Instance は、少なくとも一つの NVDEC および NVJPG ユニットを受け取れる。

さらに、H100 MIG Instance は NVIDIA 開発者ツールと連携する独自のパフォーマンスモニター群も備えるようになった。Hopper の同時プロファイリングにより、管理者は適正規模の GPU アクセラレーションを監視し、ユーザー間でリソースをシームレスに最適配分できる。

<span id="section-4-6"></span>

### 4.6 Transformer Engine

Transformer モデルは BERT から GPT-3 まで、現在広く用いられる言語モデルの基盤であり、膨大な計算リソースを必要とする。Transformer は当初、自然言語処理（NLP）向けに開発されたが、コンピュータービジョン、創薬など多様な分野への応用が進んでいる。その規模は指数関数的に増え続け、現在では数兆パラメーターに達し、トレーニング期間が数か月まで延びている。これは計算要件が大きいため、ビジネス上の要求には現実的でない。たとえば Megatron Turing NLG（MT-NLG）のトレーニングには、2,048 基の NVIDIA A100 GPU を 8 週間稼働させる必要がある。全体として、Transformer モデルは過去 5 年にわたり 2 年ごとに 275 倍というペースで、ほかのほとんどの AI モデルよりはるかに速く成長している（[図 24](#figure-24)を参照）。

<span id="figure-24"></span>

![ユースケース別に見た Transformer モデル規模の成長](../../papers/nvidia-h100-architecture/figure-24.png)

**図 24。** 異なるユースケースで指数関数的に増大する Transformer モデルの規模

H100 は、Transformer 向け AI 計算を劇的に高速化するカスタム Hopper Tensor Core 技術である、新しい Transformer Engine を搭載する。

<span id="figure-25"></span>

![Transformer Engine の動作概念](../../papers/nvidia-h100-architecture/figure-25.png)

**図 25。** Transformer Engine の動作概念。

混合精度の目標は、精度を維持するように精度形式をインテリジェントに管理しながら、より小さく高速な数値形式の性能を得ることである。Transformer モデルの各層で、Transformer Engine は Tensor Core が生成した出力値の統計を分析する。次に来るニューラルネットワーク層の種類と必要な精度を把握したうえで、Transformer Engine はテンソルをメモリへ保存する前に変換すべき目標形式も決定する。FP8 はほかの数値形式より表現範囲が狭い。利用可能な範囲を最適に使うため、Transformer Engine はテンソル統計から計算したスケーリング係数を用い、テンソルデータを表現可能な範囲へ動的にスケーリングする。したがって、各層は必要とする範囲を厳密に用いて動作し、最適な形で高速化される。

<span id="section-4-7"></span>

### 4.7 第 4 世代 NVLink と NVLink Network

超人的な対話型 AI などのタスクに向けた新しいクラスのエクサスケール HPC と 1 兆パラメーター AI モデルは、スーパーコンピューター上でもトレーニングに数か月を要する。この長いトレーニング期間を数か月から数日に短縮してビジネスでの有用性を高めるには、サーバークラスター内のすべての GPU 間で高速かつシームレスな通信が必要である。PCIe は帯域幅が限られるためボトルネックとなる。最も強力なエンドツーエンドのコンピューティングプラットフォームを構築するには、より高速で拡張性の高い NVLink インターコネクトが必要である。

NVLink は NVIDIA の高帯域幅、高エネルギー効率、低レイテンシー、ロスレスの GPU 間インターコネクトであり、データ伝送の成功を保証するリンクレベルのエラー検出やパケット再送機構などの耐障害性機能を備える。H100 GPU に実装された新しい第 4 世代 NVLink は、NVIDIA A100 Tensor Core GPU で用いられる前世代の第 3 世代 NVLink と比べて、通信帯域幅を 1.5 倍に高める。

マルチ GPU I/O と共有メモリアクセスで合計 900 GB/sec の帯域幅で動作する新しい NVLink は、PCIe Gen 5 の 7 倍の帯域幅を提供する。A100 GPU の第 3 世代 NVLink は各方向に 4 本の差動ペア（4 レーン）を用いて単一リンクを構成し、各方向で 25 GB/sec の実効帯域幅を提供するのに対し、第 4 世代 NVLink は各方向に 2 本の高速差動ペアだけを用いて単一リンクを構成し、同じく各方向で 25 GB/sec の実効帯域幅を提供する。H100 は 18 本の第 4 世代 NVLink リンクを備え、合計 900 GB/sec の帯域幅を提供する一方、A100 は 12 本の第 3 世代 NVLink リンクを備え、合計 600 GB/sec の帯域幅を提供する。

H100 は第 4 世代 NVLink に加え、複数の計算ノードにまたがる最大 256 基の GPU 間通信を可能にする、拡張可能な NVLink 版である新しい NVLink Network インターコネクトも導入する。

すべての GPU が共通のアドレス空間を共有し、GPU の物理アドレスを用いて要求が直接ルーティングされる通常の NVLink とは異なり、NVLink Network は H100 の新しいアドレス変換ハードウェアが支える新たな Network Address Space を導入し、すべての GPU のアドレス空間を相互に、またネットワークアドレス空間から隔離する。これにより、NVLink Network はより多数の GPU へ安全に拡張できる。

NVLink Network のエンドポイントは共通のメモリアドレス空間を共有しないため、NVLink Network 接続はシステム全体へ自動的に確立されない。代わりに InfiniBand などのほかのネットワークインターフェースと同様、ユーザーソフトウェアが必要に応じてエンドポイント間の接続を明示的に確立する必要がある。

<span id="section-4-7-1"></span>

#### 4.7.1 第 3 世代 NVSwitch

新しい第 3 世代 NVSwitch 技術には、サーバー、クラスター、データセンター環境で複数の GPU を接続するため、ノード内外に配置されるスイッチが含まれる。ノード内の新しい各第 3 世代 NVSwitch は、第 4 世代 NVLink リンク用に 64 ポートを提供し、マルチ GPU 接続を高速化する。スイッチの総スループットは前世代の 7.2 Tbits/sec から 13.6 Tbits/sec に増加する。

新しい第 3 世代 NVSwitch は、マルチキャストと [NVIDIA SHARP](https://docs.nvidia.com/networking/display/SHARPv200) のネットワーク内リダクションを用いる集合演算のハードウェアアクセラレーションも提供する。高速化される集合演算には、書き込みブロードキャスト（all_gather）、reduce_scatter、broadcast atomics が含まれる。ファブリック内マルチキャストとリダクションは、A100 上で [NCCL（NVIDIA Collective Communications Library）](https://developer.nvidia.com/nccl)を使用する場合と比べて、小さいブロックサイズの集合演算でレイテンシーを大幅に削減しながら、最大 2 倍のスループット向上を実現する。NVSwitch による集合演算の高速化は、集合通信が SM にかける負荷を大幅に削減する。

<span id="section-4-7-2"></span>

#### 4.7.2 新しい NVLink Switch System

新しい NVLINK Network 技術と新しい第 3 世代 NVSwitch を組み合わせることで、NVIDIA は前例のない通信帯域幅を備えた大規模なスケールアップ NVLink Switch System ネットワークを構築できる。各 GPU ノードは、ノード内 GPU の全 NVLink 帯域幅を 2:1 にテーパーしたレベルで公開する。ノードは計算ノード外に配置された NVLink Switch モジュール内の第 2 レベル NVSwitch を介して相互接続され、このスイッチが複数のノードを接続する。

NVLink Switch System は最大 256 基の GPU をサポートする。接続されたノードは 57.6 TBs の全対全帯域幅を提供し、驚異的な 1 exaFLOP の FP8 スパース AI 演算能力を供給できる。A100 と H100 に基づく 32 ノード、256 GPU の DGX SuperPOD の比較については[図 26](#figure-26)を参照されたい。[図 26](#figure-26)の性能値は参考目的に限られることに注意されたい。NVLink Switch System 技術は現在 H100 システムでは利用できないが、システムと提供時期は今後発表される予定である。

<span id="figure-26"></span>

![DGX A100 と DGX H100 の 256 GPU SuperPOD 比較](../../papers/nvidia-h100-architecture/figure-26.png)

**図 26。** DGX A100 と DGX H100 の 32 ノード、256 GPU NVIDIA SuperPOD の比較。DGX H100 SuperPOD は最大 256 基の GPU に拡張でき、第 3 世代 NVSwitch 技術に基づく新しい NVLink Switch を用いた NVLink Switch System で完全接続される。2:1 にテーパーされた fat tree トポロジーの NVLink Network インターコネクトは、たとえば全対全交換で二分帯域幅を驚異的な 9 倍に高め、前世代 InfiniBand システムに対して allreduce スループットを 4.5 倍にする。この図の性能値は参考目的に限られることに注意されたい。NVLink Switch System 技術は現在 H100 システムでは利用できないが、システムと提供時期は今後発表される予定である。

スイッチ間の最大ケーブル長は 5 m から 20 m に延長される。NVIDIA 製の OSFP（Octal Small Form Factor Pluggable）LinkX ケーブルが新たにサポートされる。OSFP あたり Quad-Port 光トランシーバーと、100G PAM4 信号の 8 チャネルを備える。Quad-Port OSFP トランシーバーの革新により、1 RU、32 ケージの単一 NVLink Switch に合計 128 個の NVLink ポートを収容でき、各ポートは 25 GB/sec でデータを転送する。

<span id="section-4-7-3"></span>

#### 4.7.3 PCIe Gen 5

H100 は PCI Express Gen 5 x16 レーンインターフェースを搭載し、A100 に含まれる Gen 4 PCIe の合計 64 GB/sec（各方向 32 GB/sec）に対して、合計 128 GB/sec（各方向 64 GB/sec）の帯域幅を提供する。

H100 は PCIe Gen 5 インターフェースを用い、最高性能の x86 CPU および [SmartNIC／DPU（Data Processing Unit）](https://www.nvidia.com/en-us/networking/products/data-processing-unit/)と接続できる。H100 は、400 Gb/s Ethernet または NDR（Next Data Rate）400 Gb/s InfiniBand によって安全な HPC および AI ワークロードのネットワークを高速化する NVIDIA BlueField-3 DPU との最適な接続を目的に設計されている。

H100 は、32 ビットおよび 64 ビットデータ型向けの atomic CAS、atomic exchange、atomic fetch add など、ネイティブ PCIe アトミック演算への対応を追加し、CPU と GPU 間の同期およびアトミック演算を高速化する。H100 は、単一の PCIe 接続 GPU を複数のプロセスまたは仮想マシン（VM）で共有・仮想化できる Single Root Input/Output Virtualization（SR-IOV）にも対応する。また H100 では、単一の SR-IOV PCIe 接続 GPU の Virtual Function（VF）または Physical Function（PF）が NVLink 経由でピア GPU にアクセスできる。

<span id="section-5"></span>

## 5 セキュリティ強化と Confidential Computing

NVIDIA は、セキュリティに敏感な市場へ販売する GPU を増やしている。Cloud Service Provider（CSP）、自動車メーカー、国立研究所、医療、金融、そのほか多くの産業や組織が高度なセキュリティを求めている。NVIDIA GPU は新しい世代ごとにセキュリティ機能を改善し続けている。

機密データが日々大量に生成、保存、処理され、規制とサイバー攻撃によるビジネスリスクが高まっている。ストレージに保存中のデータやネットワーク転送中のデータを保護する高度な暗号化技術は存在するものの、処理中または使用中のデータ保護には現在、大きな空白がある。新しい Confidential Computing 技術は、使用中のデータとアプリケーションを保護し、機密データや規制対象データを管理する組織のセキュリティを高めることで、この空白に対処する。

NVIDIA H100 は GPU の内容へのアクセスを制限して許可された主体だけがアクセスできるようにする多数のセキュリティ機能を備え、セキュアブートとアテステーション機能を提供し、システム稼働中は攻撃を能動的に監視する。さらに、専用オンチップセキュリティプロセッサー、複数の種類とレベルの暗号化への対応、ハードウェアで保護されたメモリ領域、特権アクセス制御レジスタ、オンダイセンサー、そのほか多数の機能により、顧客とそのデータに安全な GPU 処理を提供する。

H100 は Confidential Computing 機能を備えた世界初の GPU である。ユーザーは H100 GPU の前例のないアクセラレーションを利用しながら、「使用中」のデータとアプリケーションの機密性および完全性を保護できる。H100 はこのほか、ユーザーデータを保護し、ハードウェアおよびソフトウェア攻撃を防ぎ、仮想化環境と MIG 環境で VM を相互に隔離・保護する能力を高める、幅広いセキュリティ機能を提供する。

NVIDIA H100 GPU の包括的なセキュリティ機能の主な目標は次のとおりである。

- **データの保護と隔離：** 許可されていない主体が別のユーザーのデータへアクセスすることを防ぐ。主体には、ユーザー、OS、ハイパーバイザー、GPU ファームウェアが含まれうる。
- **コンテンツ保護：** GPU に保存された、または GPU で処理される保護対象コンテンツへ、許可されていない主体がアクセスすることを防ぐ。
- **物理的損傷の防止：** 悪意ある行為者によるものか事故によるものかにかかわらず、GPU の物理的損傷を防ぐ。

<span id="section-5-1"></span>

### 5.1 NVIDIA Confidential Computing

NVIDIA は [Confidential Computing Consortium（C3）](https://confidentialcomputing.io/)のメンバーである。C3 は、パブリッククラウドサービス、オンプレミスのデータセンター、エッジシステムおよびデバイスにわたり、セキュリティ脅威を減らして使用中の機密データとアプリケーションを保護する取り組みや技術の開発に協力する、世界各国のベンダー、学術機関、オープンソースプロジェクト、ソフトウェア開発者で構成される。

Confidential Computing という用語の正式な定義は、「ハードウェアベースの Trusted Execution Environment（TEE）で計算を実行することによる、使用中データの保護」である。この定義は、データがクラウド、エンドユーザーデバイス、またはその中間のどこで使用されるかに依存しない。また、どのプロセッサーがデータを保護するか、どの保護技術を使用するかにも依存しない。C3 は TEE を「データの機密性、データの完全性、コードの完全性という三つの主要特性について、一定水準の保証を提供する環境」と定義している。

今日、データは保存時にはストレージ内で、転送時にはネットワーク上で保護されることが多いが、使用中は OS／ハイパーバイザーから保護されていない。OS／ハイパーバイザーを信頼しなければならないという要件は、ユーザーのデータとコードの保護に大きな空白を残す。さらに、従来のコンピューティングインフラストラクチャでは、使用中のデータとコードを保護する能力が限られている。個人識別情報（PII）、金融・医療データなどの機密データを扱う組織や、データローカライゼーション規制への対応が必要な組織は、あらゆる段階でアプリケーション、モデル、データの機密性と完全性を狙う脅威を軽減する必要がある。

<span id="figure-27"></span>

![クラウド、オンプレミス、エッジ展開にまたがる Confidential Computing](../../papers/nvidia-h100-architecture/figure-27.png)

**図 27。** Confidential Computing は複数の ISV シナリオを保護する。Confidential Computing は、クラウド、オンプレミス、エッジにおける ISV 顧客データとトレーニング済み AI モデルの機密性を保護する。

既存の Confidential Computing ソリューションは CPU ベースであり、AI や HPC のような計算集約型ワークロードには遅すぎた。CPU ベースの Confidential Computing は一般にシステム性能を低下させるため、生産性に影響したり、レイテンシーに敏感なデータ処理ワークロードでは実用にならなかったりする。

NVIDIA Hopper アーキテクチャで導入された新しいセキュリティ機能、NVIDIA Confidential Computing により、H100 は使用中のデータとコードの両方について機密性と完全性を保護できる世界初の GPU となる。H100 はアクセラレーテッドコンピューティングを Confidential Computing の世界へ導入し、CPU の Trusted Execution Environment を GPU へ拡張する。H100 は、使用中のデータとコードを保護する必要があり、従来の Confidential Computing ソリューションでは多くのワークロードに十分な性能や柔軟性がなかったため、共有インフラストラクチャ（クラウド、コロケーション、エッジ）を利用できなかった多くのユースケースに道を開く。

NVIDIA Confidential Computing は、単一の H100 GPU、ノード内の複数の H100 GPU、または安全に保護された個々の Multi-Instance GPU（MIG）インスタンスで実行されるワークロード全体を保護・隔離する、ハードウェアベースの Trusted Execution Environment（TEE）を構築する。Trusted Execution Environment（TEE）は、GPU 上の Confidential VM と CPU 上の対応する VM の間に安全なチャネルを確立する。TEE は二つの動作モードを提供する。

1. GPU 全体を単一の VM に排他的に割り当てる（単一 VM へ複数の GPU を同時に割り当てることもできる）。
2. NVIDIA H100 GPU を分割し、MIG 技術を用いて複数の VM をサポートすることで、マルチテナント Confidential Computing を実現する。GPU アクセラレーテッドアプリケーションは TEE 内で変更せずに実行でき、手動で分割する必要はない。

ユーザーは、AI および HPC 向け NVIDIA ソフトウェアの豊富なポートフォリオと能力を、NVIDIA Confidential Computing が提供する Hardware Root of Trust のセキュリティと組み合わせ、GPU アーキテクチャの最下層でセキュリティとデータ保護を実現できる。共有またはリモートのインフラストラクチャ上でアプリケーションを実行してアテステーションを行い、ハイパーバイザー、ホスト OS、システム管理者、インフラストラクチャ所有者、物理的にアクセスできる者を含め、許可されていない主体が TEE 内で使用中のアプリケーションコードやデータを閲覧・変更できないことを保証できる。

<span id="figure-28"></span>

![さまざまなユースケース向け Confidential Computing](../../papers/nvidia-h100-architecture/figure-28.png)

**図 28。** さまざまなユースケース向け Confidential Computing

Hopper アーキテクチャの Confidential Computing 機能は、Federated Learning のような複数当事者による協調計算ユースケースのセキュリティをさらに強化・高速化する。Federated Learning により、複数の組織は各グループ固有のデータセットを共有することなく、共同で AI モデルをトレーニングまたは評価できる。H100 を用いた Confidential Federated Learning は、参加する各拠点でデータと AI モデルを外部または内部の脅威による不正アクセスから保護し、各拠点がほかの参加者で動作するソフトウェアを把握してアテステーションできるようにする。これにより、安全な協力への信頼が高まり、セキュリティ、プライバシー、規制遵守を維持しながら、医学研究の進歩、医薬品開発の迅速化、保険・金融詐欺の軽減、そのほか多くの用途が促進される。

<span id="figure-29"></span>

![3 拠点にまたがる Confidential Federated Learning](../../papers/nvidia-h100-architecture/figure-29.png)

**図 29。** Confidential Federated Learning

GPU で Confidential Computing 機能を提供するには多くのコンポーネントが関与するが、より重要な機能の一つが、以下で説明する Secure and Measured Boot である。

<span id="section-5-2"></span>

### 5.2 成功の尺度

NVIDIA Ampere GPU アーキテクチャは Secure Boot 技術を搭載していたが、Confidential Computing の要件を満たすために必要な Measured Boot には対応していなかった。ここでは H100 に実装された Secure Boot と Measured Boot の概念およびコンポーネントを簡潔に説明する。

Secure Boot とは、GPU の起動中に、NVIDIA が作成・検証した認証済みのファームウェアとマイクロコードだけを実行できる既知の安全な状態から GPU が起動されることを保証する、ハードウェアおよびソフトウェアシステムの集合である。Measured Boot は、GPU の安全な状態を決定する起動プロセスの特性を収集し、安全に保存して報告するプロセスである。アテステーションと検証は、測定値を参照値と比較し、デバイスが期待される安全な状態にあることを確認する手段である。NVIDIA はアテスター、参照値、エンドースメント署名を提供する。

デプロイワークフローは、Measured Boot で得た測定値を NVIDIA またはサービスプロバイダーの参照値と比較し、システムが顧客データの処理を開始できる準備済みかつ安全な状態にあるかを判断する。システムが検証されると、顧客は非機密コンピューティング環境で同じアプリケーションを実行する場合と同じように、アプリケーションを起動できる。

<span id="section-5-3"></span>

### 5.3 NVIDIA Confidential Computing の実装概要

[図 30](#figure-30)に示すように、左側の NVIDIA CC Off は、ホスト OS とハイパーバイザーが GPU などのデバイスへ完全にアクセスできる従来の PC アーキテクチャを示す。右側の NVIDIA CC On は、ほかの要素から VM が完全に隔離されている状態を示す。

<span id="figure-30"></span>

![NVIDIA CC Off と CC On における VM の隔離](../../papers/nvidia-h100-architecture/figure-30.png)

**図 30。** NVIDIA CC Off と CC On における VM 隔離の比較

完全な VM TEE と GPU TEE の隔離によって機密コンピューティング環境を形成する機能は、前述の内容と一部重なる次の三つの主要要素を含む、強力なハードウェアベースのセキュリティによって提供される。

- **On-Die Root of Trust（RoT）** - OS が GPU と通信する前に、GPU は RoT を用い、デバイス上で動作するファームウェアが真正で、デバイス所有者（CSP など）によって改ざんされていないことを確認する。
- **Device Attestation** - Confidential Computing が有効な真正の NVIDIA GPU と通信しており、GPU のセキュリティ状態がファームウェアやハードウェア構成を含む既知の信頼できる安全な状態と一致することを、ユーザーが確認できるようにする。
- **AES-GCM 256** - CPU と H100 GPU 間のデータ転送は、AES256-GCM のハードウェア実装を用い、PCIe のラインレートで暗号化・復号される。これにより、CPU と GPU の TEE だけが利用できる鍵を用いて、バスを介して転送されるデータの機密性と完全性の両方が確保される。この暗号実装は FIPS 140-3 level 2 の認証を受ける予定である。

NVIDIA Confidential Computing 技術を利用するために CUDA アプリケーションコードを変更する必要はない。

<span id="section-6"></span>

## 6 H100 の動画／I/O 機能

<span id="section-6-1"></span>

### 6.1 DL 向け NVDEC

H100 は A100 と比べて動画デコード能力を大幅に向上させている。DL プラットフォームでは、入力動画は H264／HEVC／VP9 など、いずれかの業界標準で圧縮されている。DL プラットフォームで高いエンドツーエンドスループットを達成する際の大きな課題の一つは、動画デコード性能とトレーニングおよび推論性能のバランスを取ることである。そうしなければ、GPU の DL 性能を完全には活用できない。H100 は、A100 の 5 基の NVDEC ユニットに対して 7 基の NVDEC（NVidia DECode）ユニットをサポートし、デコードスループットを大幅に向上させる。これにより MIG 動作時にも、各 MIG パーティションへ少なくとも 1 基の NVDEC ユニットを割り当てられる。

<span id="table-05"></span>

![A100 と H100 の動画デコードストリーム数の比較](../../papers/nvidia-h100-architecture/table-05.png)

**表 5。** A100 と H100 の動画デコード比較（ストリーム数）：

<span id="table-06"></span>

![H100 のハードウェアデコード対応](../../papers/nvidia-h100-architecture/table-06.png)

**表 6。** H100 のハードウェアデコード対応

<span id="section-6-2"></span>

### 6.2 NVJPG（JPEG）デコード

画像を対象とする DL のトレーニングと推論で高スループットを達成する際の根本的なボトルネックの一つが、画像の JPEG デコード処理（圧縮データ -> RAW データ）である。画像ビットの処理に逐次操作を用いるため、CPU と GPU は JPEG デコードの効率があまり高くない。また、CPU で JPEG デコードを行う場合は、PCIe がもう一つのボトルネックとなる。

H100 は、A100 の 1 基の 5 コアエンジンに対して、JPEG デコードを高速化する 7 基のシングルコア NVJPG HW エンジンを搭載する。

H100 NVJPG エンジンの主な特長：

- NVJPG は YUV420、YUV422、YUV444、YUV400、RGBA 形式をサポートする。
- A100 から改良された JPEG アーキテクチャ：A100 の 5 コアエンジンに代わり、H100 は 7 基のシングルコアエンジンを追加する。JPEG 画像を 5 枚ずつのバッチにまとめる代わりに個々のエンジンへ独立して割り当てられるため、ソフトウェアの利用モデルが大幅に簡素化される。また、同じバッチに異なる解像度の画像が含まれる場合のスループットも向上する。
- MIG 動作時には、各 MIG パーティションへ少なくとも 1 基の NVJPG エンジンを割り当てられる。
- JPEG スループットは A100 より大幅に向上する。

<span id="table-07"></span>

![H100 と A100 の NVJPG デコード性能](../../papers/nvidia-h100-architecture/table-07.png)

**表 7。** NVJPG デコード性能

NVIDIA は、NVDEC／NVJPG を自動的に呼び出し、動画／画像パイプラインのハードウェアアクセラレーションを管理するデータローディングライブラリ（DALI）を提供している。これにより AI 開発者は、DL ワークロードで動画／画像用ハードウェアエンジンを簡単に利用できる。また、柔軟なグラフによってカスタム動画／画像処理パイプラインを作成できる。DALI の詳細な説明とユーザーガイドは [https://docs.nvidia.com/deeplearning/dali/user-guide/docs/](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/) で参照できる。DALI ライブラリは [https://github.com/NVIDIA/DALI](https://github.com/NVIDIA/DALI) からダウンロードできる。

<span id="section-7"></span>

## 7 付録 A - NVIDIA DGX - データセンター AI の基盤ビルディングブロック

人工知能（AI）は今や、困難なビジネス課題を解決するための定番のアプローチである。顧客サービスの改善、サプライチェーンの最適化、ビジネスインテリジェンスの抽出、ほぼあらゆる産業における最先端の製品やサービスの設計など、AI は組織がイノベーションを実現するための仕組みを提供する。そして AI インフラストラクチャの先駆者として、NVIDIA DGX システムは、こうした不可欠なアイデアを実現するための最も強力で完全な AI プラットフォームを提供する。

<span id="section-7-1"></span>

### 7.1 NVIDIA DGX H100 - 世界で最も完全な AI プラットフォーム

NVIDIA DGX H100 はビジネスのイノベーションと最適化を支える。NVIDIA の名高い DGX システムの最新版であり、NVIDIA DGX SuperPOD の基盤でもある DGX H100 は、画期的な NVIDIA H100 Tensor Core GPU を搭載した AI のパワーハウスである。このシステムは AI スループットを最大化するという唯一の目的のために設計されており、自然言語処理、レコメンダーシステム、データ分析などで企業がブレークスルーを達成できるよう、高度に洗練され、体系化され、拡張可能なプラットフォームを提供する。オンプレミスのほか、多様なアクセスおよびデプロイ方法で利用できる DGX H100 は、企業が AI によって最大の課題を解決するために必要な性能を提供する。

<span id="section-7-2"></span>

### 7.2 DGX H100 の概要

NVIDIA DGX H100 は、トレーニング、推論、分析に対応する汎用高性能 AI システムである。DGX H100 は Bluefield-3、NDR InfiniBand、第 2 世代 MIG 技術を搭載し、クラウドネイティブに対応する。単一の DGX H100 システムは、比類のない 32 petaFLOPS の性能を発揮する。複数の DGX H100 システムを DGX POD、さらに DGX SuperPOD と呼ばれるクラスターへ接続することで、この性能を容易にスケールアップできる。

各 DGX H100 システムは次の要素で構成される。

- H100 Tensor Core GPU x 8
- 第 4 世代 Tensor Core
- 第 4 世代 NVLink
- 第 3 世代 NVSwitch（x4）
- ConnectX-7（400 Gb/s InfiniBand／Ethernet）x8
- Bluefield-3 DPU x2
- PCIe Gen5 対応

<span id="section-7-3"></span>

### 7.3 比類のないデータセンター拡張性

NVIDIA DGX H100 は、拡張可能な AI インフラストラクチャの企業向け設計図である [NVIDIA DGX SuperPOD](https://www.nvidia.com/en-us/data-center/resources/nvidia-dgx-superpod-reference-architecture/) など、大規模 AI クラスターの基盤ビルディングブロックである。DGX H100 の 8 基の NVIDIA H100 GPU は、新しい高性能な第 4 世代 NVLink 技術を用い、4 基の第 3 世代 NVSwitch を介して相互接続される。第 4 世代 NVLink 技術は前世代の 1.5 倍の通信帯域幅を提供し、PCIe Gen5 より最大 7 倍高速である。GPU 間の総スループットは最大 7.2 TB/sec で、前世代の DGX A100 と比べて約 1.5 倍向上する。各 400 Gb/sec で動作する 8 基の NVIDIA ConnectX-7 InfiniBand／Ethernet アダプターも組み合わせることで、DGX H100 システムは大規模 AI ワークロード向けの強力な高速ファブリックを提供する。

各 DGX H100 は、ストレージ、セキュリティ、ネットワーク管理機能をインテリジェントにハードウェア高速化する NVIDIA BlueField-3 DPU（Data Processing Unit）も 2 基搭載する。BlueField-3 DPU は、従来のコンピューティング環境を安全で高速化された仮想プライベートクラウドへ変革し、組織が安全なマルチテナント環境でアプリケーションワークロードを実行できるようにする。BlueField-3 は、データセンターインフラストラクチャをビジネスアプリケーションから分離し、データセンターのセキュリティを強化し、運用を合理化して総所有コストを削減する。NVIDIA のネットワーク内コンピューティング技術を備えた BlueField-3 は、次世代スーパーコンピューティングプラットフォームを実現し、最適なベアメタル性能とマルチノードのテナント隔離へのネイティブ対応を提供する。

大規模な GPU アクセラレーテッドコンピューティング、最先端のネットワークハードウェア、ソフトウェア最適化を組み合わせることで、NVIDIA DGX H100 は数百または数千ノードへ拡張し、次世代 AI アプリケーションの最大の課題に対応できる。

<span id="section-7-4"></span>

### 7.4 NVIDIA DGX H100 システム仕様

<span id="table-08"></span>

![NVIDIA DGX H100 と DGX A100 のシステム仕様比較](../../papers/nvidia-h100-architecture/table-08.png)

**表 8。** NVIDIA DGX H100 システム仕様

<span id="section-8"></span>

## 8 付録 B - NVIDIA CUDA プラットフォームの更新

[NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit) は、包括的で生産性と性能に優れたアクセラレーテッドコンピューティングプラットフォームである。GPU、CPU、DPU、ネットワーク内コンピューティングを用い、システムソフトウェアからアプリケーション固有のライブラリとフレームワークまで、あらゆるレベルのエンドユーザーアプリケーションを高速化する（[図 31](#figure-31)を参照）。成熟して使いやすいツールチェーン、開発者ツール、ドキュメントは、アクセラレーテッド異種アプリケーションに最高の開発者体験を提供する。

<span id="section-8-1"></span>

### 8.1 高性能ライブラリとフレームワーク

CUDA ライブラリは、一般的な数学演算（[CUDA Math Library](https://developer.nvidia.com/cuda-math-library)）、並列アルゴリズム（[CUB](https://github.com/NVIDIA/cub) および [Thrust](https://developer.nvidia.com/thrust)）、線形代数（[cuBLAS](https://developer.nvidia.com/cublas)）、密およびスパース線形ソルバー（[cuSOLVER](https://developer.nvidia.com/cusolver) および [cuSPARSE](https://developer.nvidia.com/cusparse)）、FFT（[cuFFT](https://developer.nvidia.com/cufft)）、乱数生成（[cuRAND](https://developer.nvidia.com/curand)）、テンソル操作（[cuTENSOR](https://developer.nvidia.com/cutensor)）、画像・信号処理（[NPP](https://developer.nvidia.com/npp)）、JPEG デコード（[nvJPEG](https://developer.nvidia.com/nvjpeg)）、GPU 管理（[NVML](https://developer.nvidia.com/nvidia-management-library-nvml)）の性能を最大化する。[cuNumeric](https://developer.nvidia.com/cunumeric) はコードを一切変更せず、Legate と Legion ランタイムを介して NumPy プログラムを透過的に高速化し、あらゆる規模のマシンへ分散する。[libcu++](https://nvidia.github.io/libcudacxx/) は、高度に並行で異種構成かつ ISO 標準準拠の C++ アプリケーションを実現するため、異種環境向けの同期およびデータ移動プリミティブを提供する。

さらに、CUDA プラットフォームの通信ライブラリは、標準に基づく拡張可能なシステムプログラミングを可能にする。[HPC-X](https://developer.nvidia.com/networking/hpc-x) は CUDA 対応 MPI ライブラリで、RDMA を用いて GPU バッファーを直接送受信する GPUDirect をサポートする。[NVIDIA Collective Communications Library（NCCL）](https://developer.nvidia.com/nccl)は、高度に最適化されたマルチノード集合通信プリミティブを実装する。[NVSHMEM](https://developer.nvidia.com/nvshmem) は OpenSHMEM に基づき、ホストスレッドとデバイススレッドの両方に異種マルチノード通信プリミティブを提供する。[cuFile](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html) と [MAGNUM IO](https://developer.nvidia.com/magnum-io) は、[GPUDirect Storage](https://developer.nvidia.com/gpudirect-storage) を介して高性能なファイル I/O を実現し、異種アプリケーションを可能にする。

さらに、領域固有のライブラリとフレームワークからなる広範なスイートが、多様な応用分野の主要アルゴリズムを高速化する。たとえば、ディープニューラルネットワーク（[cuDNN](https://developer.nvidia.com/cudnn)）、シミュレーションおよび陰的非構造手法向け線形ソルバー（[AmgX](https://developer.nvidia.com/amgx)）、量子コンピューティング（[cuQuantum](https://developer.nvidia.com/cuquantum-sdk)）、データサイエンスと機械学習（[RAPIDS](https://rapids.ai/)）、機械学習向けデータローディングと前処理（[DALI](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/)）、リアルタイム 3D シミュレーションと設計コラボレーション（[Omniverse](https://developer.nvidia.com/nvidia-omniverse-platform)）などがある。150 を超える [Software Development Kit](https://developer.nvidia.com/) がこれらのライブラリを活用し、ハイパフォーマンスコンピューティング（[NVIDIA HPC SDK](https://developer.nvidia.com/hpc-sdk)）、AI、[Machine Learning](https://developer.nvidia.com/machine-learning)、[Deep Learning](https://developer.nvidia.com/deep-learning)、Data Science、ゲノミクス（[NVIDIA CLARA](https://developer.nvidia.com/clara)）、スマートシティ（[NVIDIA Metropolis](https://developer.nvidia.com/metropolis)）、自動運転（[NVIDIA Drive SDK](https://developer.nvidia.com/drive)）、通信（[NVIDIA Aerial SDK](https://developer.nvidia.com/aerial-sdk)）、ロボティクス（[NVIDIA Isaac SDK](https://developer.nvidia.com/isaac-sdk)）、サイバーセキュリティ（[NVIDIA Morpheus SDK](https://developer.nvidia.com/morpheus-cybersecurity)）、[Computer Vision](https://developer.nvidia.com/computer-vision) など、幅広い応用分野で開発者が高い生産性を発揮できるよう支援する。

<span id="figure-31"></span>

![NVIDIA CUDA プラットフォームとそのエコシステム](../../papers/nvidia-h100-architecture/figure-31.png)

**図 31。** NVIDIA CUDA プラットフォームとそのエコシステム

<span id="section-8-2"></span>

### 8.2 システムソフトウェア

NVIDIA CUDA プラットフォームは、大規模な異種システムをユーザーが生産的かつ効率的にデプロイ、管理、最適化するのに役立つ、柔軟なシステムソフトウェアコンポーネントも提供する。その提供範囲は、デバイスドライバー（CUDA ドライバー）、デバイス管理ソフトウェア（NVML、NVIDIA-smi、DCGM、Unified Fabric Manager）、異種ネットワークおよびファイル I/O 向け GPUDirect から、コンテナ対応のジョブスケジューリングシステムとオペレーティングシステム（DGX OS）に及ぶ。

<span id="section-8-3"></span>

### 8.3 ドキュメントとトレーニング

大規模な CUDA ソフトウェアエコシステムは、C++ 並列アルゴリズムなどのプログラミングモデル、libcu++ などのライブラリ、RAPIDS AI などのフレームワーク、HPC SDK などの SDK に向けた優れたドキュメントによって補完される。

NVIDIA Deep Learning Institute（DLI）は、たとえば Supercomputing や International Supercomputing Conference といったカンファレンスで、自習型およびライブ形式のトレーニングを提供し、個人が AI、アクセラレーテッドコンピューティング、アクセラレーテッドデータサイエンス、グラフィックスとシミュレーションなどの知識を深められるようにする。DLI は、研究機関や HPC センターの有資格教育者を DLI Ambassador としてトレーニング・認定し、それぞれのニーズに合わせて DLI コンテンツを教え、調整できるようにする。

公式ドキュメントに加え、NVIDIA はさまざまなコミュニティや HPC サイトと提携し、GPU Hackathon and Bootcamp プログラムを提供している。領域科学者と Research Software Engineer（RSE）のチームを NVIDIA および HPC コミュニティの GPU メンターと組み合わせ、現代の異種コンピューティングシステムを効果的に利用するために必要なソフトウェア開発、並列コンピューティング、最適化のスキルを伝える。NVIDIA は毎年 GPU Technology Conference（GTC）を開催し、最新の NVIDIA プラットフォームと技術について開発者を教育することに重点を置いている。講演では NVIDIA のプログラミングモデル、ハードウェアの詳細、幅広い分野へのアクセラレーテッドコンピューティングの応用を扱う。これらの講演はすべて録画され、GTC on demand で視聴できる。

<span id="section-8-4"></span>

### 8.4 言語とコンパイラー

CUDA プラットフォームは、NVIDIA の NVVM IR と libNVVM を介して高度に最適化されたデバイスバイナリを生成する、統一された柔軟なコンパイラースタックを公開する。NVVM IR は LLVM 7 に基づくコンパイラー中間表現（IR）で、GPU 計算カーネルを生成するためのフロントエンドコンパイラーターゲットを提供する。libNVVM は、NVVM IR を NVIDIA GPU の仮想 ISA である PTX へコンパイルして最適化するためのライブラリである。すべての NVIDIA Compute コンパイラーは libNVVM を用いて NVIDIA GPU をターゲットとし（[図 32](#figure-32)）、ユーザーやフレームワークが CUDA C++ 自体と同じコード生成品質および最適化によって、選択したプログラミング言語を CUDA プラットフォームへ導入できるようにする。

<span id="figure-32"></span>

![libNVVM を用いる高水準言語フロントエンド](../../papers/nvidia-h100-architecture/figure-32.png)

**図 32。** 高水準言語フロントエンド。フロントエンドは libNVVM を用いて NVVM IR プログラムを PTX へコンパイルし、GPU 上で実行する。

NVIDIA GPU の仮想 ISA である PTX は、NVIDIA の対象アーキテクチャ上で効率的に実行するため、サードパーティの生成系がターゲットとする公開 ISA である。PTX には前方互換性があり、オフラインまたは実行時にアセンブルできるという利点もある。

多くのアプリケーションでは、生成する GPU 計算カーネルがプログラム入力に依存する。これらのアプリケーションは NVVM IR を生成することもできるが、NVIDIA Runtime Compiler を使えば、代わりに馴染みのある CUDA C++ を生成できるため、アプリケーションとそのユーザーの生産性が大幅に向上する。NVRTC は libNVVM を用いて CUDA C++ を実行時に PTX へコンパイルし、組み込み PTX アセンブラーを用いてネイティブ GPU バイナリコードへコンパイルすることもできる。これにより、たとえば Python プログラムなどのアプリケーションはユーザー入力に応じてプログラム用カーネルを動的に生成でき、たとえば C++ プログラムはプログラム入力に応じて実行時に計算カーネルを特殊化できる。

[NVIDIA HPC SDK](https://developer.nvidia.com/hpc-sdk) は、異種システム向けのツールチェーン群である。NVCC は、GPU コンパイルを GCC などの外部ホストコンパイラーと組み合わせる分割コンパイルモデルを提供する CUDA C++ コンパイラーである（[図 33](#figure-33)左）。NVIDIA HPC コンパイラーである NVC、NVC++、NVFortran は、統一された異種コンパイルモデルを提供する（[図 33](#figure-33)右）。

<span id="figure-33"></span>

![NVCC の分割コンパイルモデルと NVC++ の統一コンパイルモデル](../../papers/nvidia-h100-architecture/figure-33.png)

**図 33。** NVCC の分割コンパイルモデルと NVC++ の統一コンパイルモデル

統一コンパイラーは、異なるターゲット向けにコンパイル処理を分割する前に、プログラムを一度だけ解析・最適化する。このモデルでは、nvcc で利用できない特定の機能が使用できる。たとえば nvcc では、CUDA C++ デバイスコードに `__device__` アノテーションが必要である（[図 34](#figure-34)左）。NVC++ コンパイラーはこれらのアノテーションを必要とせず、プログラムが特定のターゲットから関数を使用し、その定義へ到達可能であれば、コンパイラーはその関数のコンパイルを試みる（[図 34](#figure-34)右）。

<span id="figure-34"></span>

![NVCC のデバイスアノテーションと NVC++ が推論する実行空間](../../papers/nvidia-h100-architecture/figure-34.png)

**図 34。** 統一ツールチェーンは実行空間の推論をサポートする

統一コンパイルは開発を簡素化し、初心者にとって GPU プログラミングを身近にすると同時に、経験豊富な開発者の生産性も高める。また、ホストとデバイスのターゲット間でコードを再利用しやすくなり、GPU アプリケーションを高速化するプロセスが簡素化される。

<span id="section-9"></span>

## 9 付録 C - DPX 命令によるゲノミクス高速化

NVIDIA H100 は、従来の GPU や CPU と比べてさまざまな倍率で、多種多様なアプリケーションとアルゴリズムを高速化できる。本節では、ゲノミクス分野で H100 がもたらす大幅な高速化に焦点を当てる。感染症の増加と世界的パンデミックの危険性を経験した近年ほど、ゲノムとタンパク質の解析が人類にとって重要になったことはない。

H100 は新しい DPX 命令を導入する。これは、DNA 遺伝子シーケンシングに用いる Smith-Waterman アルゴリズムや、タンパク質の分類・フォールディングなど、Dynamic Programming アルゴリズムを高速化する新しい専用ハードウェア命令である。H100 は Smith-Waterman で NVIDIA Ampere A100 GPU に対して最大 7 倍の高速化を実現し、疾病診断、ウイルス変異研究、ワクチン開発で解を得るまでの時間を大幅に短縮する。以下に、ゲノミクスと遺伝子シーケンシングの短いチュートリアルを示す。

ゲノミクス分野は指数関数的に成長し、医療、農業、生命科学の各産業を変革しているだけでなく、SARS-CoV-2 や COVID-19 と闘うための最も鋭い武器の一つにもなっている。ヒトゲノム全体または選択した一部をシーケンシングすることは、その仕組みを理解するうえで極めて重要であり、それによって疾病を引き起こす遺伝的変異、保護をもたらす遺伝的変異、治療標的となりうる遺伝的変異を特定できる。組織が疾病の理解、創薬、患者ケアの向上にゲノムを活用するなか、データ分析と管理はゲノムの価値を引き出す主要なツールとなっている。

2005 年に次世代シーケンシング（NGS）が導入されて以来、この産業ではデータが爆発的に増加し、家系の解明から臨床医療まで、ヒトゲノムを中心とした新しい産業が生まれた。ゲノミクスは、生の装置データを生物学的知見へ変換するために必要な計算集約的手順を高速化できる高度なコンピューティングシステムから恩恵を受ける。個人のゲノムの生データサイズは約 100 ギガバイト（GB）である。ディープラーニングや自然言語処理などの複雑なアルゴリズムとアプリケーションを用いる解析後には、総データフットプリントが 225 GB を超える。GPU で数学モデルを高速化すると、シーケンシングリード処理や変異同定など従来のゲノミクス解析に明確な利点が得られるだけでなく、特定のゲノム変異が疾病や健康へ及ぼす影響についての理解を一変させる可能性もある。

NVIDIA Clara™ Parabricks® は次世代シーケンシングデータ向けのアクセラレーテッドコンピューティングフレームワークであり、DNA および RNA アプリケーションのエンドツーエンドデータ分析ワークフローをサポートする。NVIDIA GPU プラットフォーム群で動作する Clara Parabricks は、GPU で高速化された Burrows-Wheeler Aligner（BWA-MEM）、Picard、Samtools など 50 を超える高速化ツールに加え、複数の Variant Call Format（VCF）へアノテーションを施し、フィルタリングして結合するユーティリティ群を提供する。ワークフロー全体にわたる高速化ツールの組み合わせにより、数時間または数日ではなく数分で結果を生成できる。

<span id="figure-35"></span>

![NVIDIA Clara Parabricks アクセラレーテッドフレームワーク](../../papers/nvidia-h100-architecture/figure-35.png)

**図 35。** NVIDIA CLARA Parabricks アクセラレーテッドフレームワーク

ゲノムとは、生物の発生とあらゆる活動の指示に必要な遺伝情報を含む化合物、デオキシリボ核酸（DNA）の完全な一式である。DNA 分子は、対をなしてねじれた 2 本の鎖で構成される。各鎖は、ヌクレオチド塩基と呼ばれる 4 種類の化学単位からなる。塩基はアデニン（A）、チミン（T）、グアニン（G）、シトシン（C）である。向かい合う鎖の塩基は特異的に対をなし、A は必ず T と、C は必ず G と対になる。ヒトゲノムには約 30 億の塩基対が含まれ、すべての細胞の核内にある 23 対の染色体に存在する。ゲノムのシーケンシングとは、DNA 断片における塩基対の正確な順序を決定することである。

個人の DNA シーケンシング処理は、DNA を相補対に分離し、DNA 鎖を特定の大きさ（100～2,000 塩基対の場合がある）の断片へ切り分け、コンピューターで読み取れる塩基対コード列を生成するシーケンシング装置で、リードと呼ばれるこれらの小断片をシーケンシングする化学処理から始まる。次に、シーケンシングした断片を、参照ゲノム内で配列の位置を検索するか、参照ゲノム配列に依存せず塩基の重複パターンを探して断片を組み立てる De Novo 法によって再構成する。

計算の観点では、この問題は数十億塩基対の参照ゲノムから一連の「リード」を検索して照合すること、または数百万のリードを比較して重複を見つけ、正しい順序に整列させるパターン照合アルゴリズムによりゲノムをゼロから組み立てることに帰着する。この処理中、アルゴリズムは不一致を解消するために配列を挿入、編集、削除し、発生しうるさまざまな種類の不一致のコストを指定する必要がある場合もある。したがって、パターン照合の計算ハードウェアアーキテクチャには、これらの要件に適応できる柔軟性に加え、タンパク質シーケンシングなどゲノミクスの別の問題で用いる、ほかの種類の類似アルゴリズムへの対応も必要である。

DNA シーケンシング向け Smith-Waterman アルゴリズムは、NVIDIA CLARA Parabricks アクセラレーテッドコンピューティングフレームワークの GPU アクセラレーテッド BWA-MEM モジュールで用いられる。このアルゴリズムは基本的に、二つの塩基リード文字列を比較してスコアリング行列を作成し、行列内のスコアをトレースバックして二つの文字列に最もよく一致するパターンを特定する。このアルゴリズムがゲノムシーケンシングでどのように使われるかについては、[こちら](https://en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm)に詳しい説明がある。

<span id="figure-36"></span>

![Smith-Waterman のスコアリング行列とトレースバック](../../papers/nvidia-h100-architecture/figure-36.png)

**図 36。** ゲノムシーケンシング向け Smith-Waterman アルゴリズム [+1]

上の図では、行列の各セルを更新するために 5 種類の基本計算が必要となる。

1. 一致時に、対角要素へ x の値（この図では x = 3）を加算する。
2. 不一致時に、対角要素から x の値を減算する。
3. 垂直要素の不一致時に、y の値（この図では y = 2）を減算する。
4. 水平要素の不一致時に、z の値（この図では z = 2）を減算する。
5. 上記四つの演算の最大値を求める（結果が負ならセルをゼロにする）。

H100 の新しい DPX 命令は、上記の計算集合とほかの類似アルゴリズムを高速化するよう最適化されている。

[+1]: 出典：[https://en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm](https://en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm)

<span id="section-10"></span>

## 10 注意事項

本仕様に記載された情報は、記載日現在において正確かつ信頼できると考えられている。ただし NVIDIA Corporation（「NVIDIA」）は、明示または黙示を問わず、当該情報の正確性または完全性についていかなる表明も保証も行わない。NVIDIA は、当該情報の結果または使用、ならびにその使用に起因しうる特許または第三者の権利の侵害について、一切の責任を負わない。本書は、以前に提供された可能性のある本製品に関するほかのすべての仕様に優先し、それらを置き換える。

NVIDIA は、本仕様をいつでも訂正、変更、強化、改善、その他の方法で変更する権利、および通知なく製品またはサービスを中止する権利を留保する。顧客は注文前に最新の関連仕様を入手し、当該情報が最新かつ完全であることを確認しなければならない。

NVIDIA 製品は、NVIDIA と顧客の権限ある代表者が署名した個別の売買契約で別段の合意がない限り、注文確認時に提示される NVIDIA の標準販売条件に従って販売される。NVIDIA は、本仕様で言及される NVIDIA 製品の購入に関して、顧客の一般取引条件を適用することに明示的に異議を唱える。

NVIDIA 製品は、医療、軍事、航空機、宇宙、生命維持装置での使用、または NVIDIA 製品の故障や誤動作が人身傷害、死亡、財産または環境への損害を引き起こすと合理的に予想される用途に適するよう設計、認可、保証されていない。NVIDIA は、当該装置または用途に NVIDIA 製品を組み込むこと、および／または使用することについて一切の責任を負わず、したがって当該組み込みおよび／または使用は顧客自身の責任で行われる。

NVIDIA は、これらの仕様に基づく製品が追加の試験や変更なしに特定の用途へ適することについて、いかなる表明も保証も行わない。各製品のすべてのパラメーターについて、NVIDIA が必ずしも試験を実施するわけではない。製品が顧客の予定する用途に適合することを確認し、アプリケーションまたは製品の不具合を避けるために必要な試験を実施することは、顧客のみの責任である。顧客の製品設計の弱点が NVIDIA 製品の品質と信頼性に影響し、本仕様に含まれる条件や要件とは別の、または追加の条件や要件につながる場合がある。NVIDIA は、次の事項に基づく、または起因しうる不具合、損害、費用、問題について、一切の責任を負わない。（i）本仕様に反する方法で NVIDIA 製品を使用すること、または（ii）顧客の製品設計。

本仕様によって、NVIDIA の特許権、著作権、その他の知的財産権に基づく明示または黙示のライセンスは付与されない。NVIDIA が公開する第三者の製品またはサービスに関する情報は、当該製品またはサービスを使用する NVIDIA からのライセンス、あるいはそれらの保証や推奨を意味しない。当該情報を使用するには、第三者の特許その他の知的財産権に基づく第三者からのライセンス、または NVIDIA の特許その他の知的財産権に基づく NVIDIA からのライセンスが必要になる場合がある。本仕様に記載された情報は、複製について NVIDIA の書面による承認を得ており、変更せずに複製し、関連するすべての条件、制限、通知を添付する場合に限り、複製できる。

NVIDIA のすべての設計仕様、リファレンスボード、ファイル、図面、診断、一覧、その他の文書（まとめて、または個別に「資料」）は「現状有姿」で提供される。NVIDIA は、資料に関して明示、黙示、法定、その他を問わず一切保証せず、非侵害性、商品性、特定目的への適合性に関するすべての黙示保証を明示的に否認する。顧客が何らかの理由で被る損害にかかわらず、本書に記載された製品に対する NVIDIA の総責任および累積責任は、当該製品に関する NVIDIA の販売条件に従って制限される。

<span id="section-10-1"></span>

### 10.1 商標

NVIDIA、NVIDIA ロゴ、NVIDIA CUDA、NVIDIA Omniverse、NVIDIA RTX、NVIDIA Tesla、NVIDIA Turing、NVIDIA Volta、NVIDIA Jetson AGX Xavier、NVIDIA DGX、NVIDIA HGX、NVIDIA EGA、NVIDIA CUDA-X、NVIDIA GPU Cloud、GeForce、Quadro、CUDA、GeForce RTX、NVIDIA NVLink、NVIDIA NVSwitch、NVIDIA DGX POD、NVIDIA DGX SuperPOD、NVIDIA TensorRT は、米国およびその他の国における NVIDIA Corporation の商標および／または登録商標である。その他の会社名および製品名は、それぞれ関係する会社の商標である場合がある。

Copyright © 2023 NVIDIA Corporation. All rights reserved.
