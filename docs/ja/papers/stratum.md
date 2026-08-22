---
title: 'Stratum'
createTime: 2026/08/22 20:00:00
permalink: /ja/papers/stratum/
pageClass: paper-reading
---

> [Yue Pan](https://dblp.org/pid/385/3702)、[Zihan Xia](https://dblp.org/pid/244/0846) [+equal contribution]、[Po-Kai Hsu](https://shimeng.ece.gatech.edu/people/)、[Lanxiang Hu](https://snyhlx.github.io/)、[Hyungyo Kim](https://cubic.engineering.columbia.edu/directory/hyungyo-kim)、[Janak Sharda](https://grad.gatech.edu/events/phd-dissertation-defense-janak-sharda)、[Minxuan Zhou](https://zhouminxuan.github.io/)、[Nam Sung Kim](https://ece.illinois.edu/about/directory/faculty/nskim)、[Shimeng Yu](https://ece.gatech.edu/directory/shimeng-yu)、[Tajana Rosing](https://cseweb.ucsd.edu/~trosing/)、[Mingu Kang](https://jacobsschool.ucsd.edu/node/3664)。2025 年 10 月 6 日に arXiv へ初版を投稿し、現在の版は v1。[Stratum: System-Hardware Co-Design with Tiered Monolithic 3D-Stackable DRAM for Efficient MoE Serving](https://arxiv.org/abs/2510.05245)。[原論文 PDF](/paper/stratum.pdf)。[DOI](https://doi.org/10.48550/arXiv.2510.05245)。[MICRO '25 DOI](https://doi.org/10.1145/3725843.3756043)。[TeX ソース](https://export.arxiv.org/e-print/2510.05245v1)。正確な印刷レイアウトと参考文献は原論文 PDF に従う。

## 概要

大規模言語モデル (LLM) の発展に伴い、Mixture of Experts (MoE) は幅広いタスクで最先端性能を実現する主流の構成になっている。MoE は疎なゲーティングにより入力ごとに少数のエキスパート・サブネットワークだけを活性化し、数十億パラメータの容量を小規模モデルに近い推論コストで実現する。一方、MoE 層が導入する膨大なデータ量はハードウェア展開を難しくする。そこで、モノリシック 3D スタッカブル DRAM (Mono3D DRAM)、近メモリ処理 (NMP)、GPU アクセラレーションを組み合わせたシステム-ハードウェア協調設計 Stratum を提案する。ロジックダイと Mono3D DRAM ダイはハイブリッドボンディングで接続し、Mono3D DRAM スタックと GPU はシリコンインタポーザで接続する。モノリシック構造による高密度な垂直接続により、Mono3D DRAM は HBM より高い内部帯域幅を持ち、高性能な近メモリ処理を可能にする。さらに、$z$ 方向の急速な垂直スケーリングで生じる層間レイテンシ差に対し、内部メモリ階層を構成し、トピック別のエキスパート使用予測に基づいてアクセス確率の高いデータを各層へ配置する。Stratum は複数のベンチマークで、GPU ベースラインに対してデコードスループットを最大 $8.29\times$、エネルギー効率を $7.66\times$ 改善する。

<span id="section-1"></span>

## 1 はじめに

Transformer ベースの大規模言語モデル (LLM) は幅広いアプリケーションの中核となり、多様な領域で最先端の性能を実現している [Vas17c, Dub24, Dos20, Zha22, Gro25a, Jia24a, Ope23b, Qwe25, Den20, Dee24a, Dee25c]。さまざまなタスクの性能を高めるため、LLM は前例のない規模へ拡大しており、LLaMA 3.1 (405B) [Dub24]、DeepSeek-V3 (671B) [Dee24a]、Kimi-K2 (1T) [Kim25e] などがモデルサイズと性能の限界を押し広げている。これらの大規模モデルの学習と展開は、特にメモリ容量と計算能力の点で、基盤インフラに大きな課題をもたらす。

推論コストを削減するさまざまな取り組みの中で、活性化の疎性を利用する方法は、計算量とデータ移動量を直接削減できる有望な解決策である。広く採用されている方法の一つが Mixture of Experts (MoE) アーキテクチャ [Dee24a, Ope23b, Olm24, Jia24a, Du21a, Dbr25, Gro25a, Fed22, Lla25] であり、[図 1](#figure-01) に示すように、従来の密な Multi-Layer Perceptron (MLP) ブロックを、推論時に疎に選択される複数のエキスパート MLP へ置き換える。MoE モデルはルーティング機構を使い、各 token で少数のエキスパートだけを活性化する。MLP がモデル全体のサイズの大部分を占めるため、この選択的な活性化は推論と学習のコストを大幅に削減する [Sca24]。その結果、MoE アーキテクチャは多くの最先端 LLM で好まれる選択肢となっている。

<span id="figure-01"></span>

![図 1. Architectures of dense transformer-based LLM (left) and Mixture of Experts (MoE) LLM (right).](../../papers/stratum/figure-01.png)

**図 1.** Architectures of dense transformer-based LLM (left) and Mixture of Experts (MoE) LLM (right).

MoE モデルは実際のメモリアクセスと計算量を削減するが、モデル全体のサイズは解決しない。モデルサイズの急速な増大には、高帯域かつ高密度なメモリ技術が必要となる。その流れで、ダイ積層 High Bandwidth Memory (HBM) は NVIDIA A100 や H100 などの高性能 GPU で主流の解決策となった [Nvi21, Nvi23b]。6 枚の DRAM ダイを積層し、1024-bit I/O インターフェースを備え、シリコンインターポーザを介して GPU 計算ダイへ 1 スタックあたり最大 800 GB/s の帯域を供給する。HBM は従来の 2D DRAM より高帯域だが、インターポーザを通る帯域はなお不足している。この制約により、特に LLM デコードのようなメモリ律速の処理で GPU 計算資源が十分に利用されない [Att24]。HBM と GPU の間のメモリウォールを緩和するため、近年の手法は LLM 推論に near-memory processing (NMP) を導入している [New20, Tra22, Har21, Att24, Dup24, Neu24a, Hig21]。先行研究 [Neu24a, Att24, Tra22, Dup24] は HBM のベースダイに計算ロジックを配置し、デコード段階の注意機構を NMP ユニットで計算した。しかし、ベースダイの NMP も、限られた TSV I/O 接続をデータが垂直に通過するため帯域が制限される。この制約を緩和するため、メモリダイに計算ユニットを直接統合して大きな内部帯域を利用する研究が行われており [Har21, Tra22, Dri17, Att24, Hig21, Pri24]、一般に processing in memory (PIM) と呼ばれる。一方、DRAM ダイに埋め込んだ計算ロジックは、メモリ内データ転送のコストと、計算よりも記憶に最適化された DRAM 技術でロジックを実装することによる大きな性能・面積・電力 (PPA) オーバーヘッドを伴う [Dri17]。さらに、同一ダイ上でロジックとメモリを統合すると、熱と製造に関する追加の懸念が生じる。

HBM の強力な代替として、本文で Mono3D DRAM と呼ぶ Monolithic 3D-Stackable DRAM が、10 nm 未満の技術でも DRAM のスケーリングを継続できる有望な解決策として登場した。高価な TSV とボンディング工程を不要にする低コストの製造工程により垂直統合を改善し、産業界と学術界の双方で注目を集めている [Ong23, A24, Sig25, Mon25]。同じウェハ上に複数の DRAM 層を順に製造することで、bit あたりのコストを比例して増やさずに高密度化でき、将来の大容量メモリシステムの有力候補となる。HBM ベース NMP と比べ、Mono3D DRAM ベース NMP には重要なアーキテクチャ上の利点がある。Mono3D DRAM は DRAM 内のモノリシック構造と DRAM ダイ・ロジックダイ間の面対面ハイブリッドボンディングにより、チップ全面を活用して大幅に高い内部帯域を提供する。一方、HBM の TSV は垂直配線としてロジックベースダイと DRAM ダイの双方に面積を必要とし、その面積を無制限には増やせないため内部帯域が制約される。さらに、1 $\mu m$ のハイブリッドボンディングピッチ [Che20a] は HBM より約 $5\times$ 微細な垂直配線ピッチであり [Exp24]、より密な内部接続を可能にする。Mono3D DRAM の高い内部帯域により、ロジックダイに実装した NMP は従来の HBM メモリダイ NMP より強力になる。モノリシック統合でダイが薄くなり垂直熱伝導も改善するため、放熱性が高まり、より高い電力密度と NMP の大きな電力予算を支えられる。

Mono3D DRAM には多くの利点があるが、その優位性を十分に引き出すには重要な課題が残る。近年の研究は、層を順に製造することで数百層の垂直積層を統合できることを示した [Sig25, Mon25]。しかし、このような積極的な垂直スケーリングは、層ごとのアクセス遅延に大きなばらつきをもたらす。最悪遅延を前提とした単純な設計では、利用可能な内部帯域を大きく損なう。また、微細ピッチのモノリシック 3D 統合によって垂直配線密度が大幅に高まり、大量のデータへ同時にアクセスできるため、ローカルな Mono3D DRAM bank の帯域を活用しつつ bank 間・channel 間アクセスを抑える、慎重に設計したデータマッピングが必要となる。さらに、ローカル DRAM のデータ帯域が極めて高いため、データのマッピングが不適切だと処理ユニット間のチップ内通信オーバーヘッドが計算遅延に匹敵する。したがって、実行時間全体を最小化するには計算と通信をバランスよくオーバーラップさせることが重要である。

大規模 MoE モデルのサービス課題に対処するため、Mono3D DRAM、NMP、GPU を統合した Stratum システムを提案する。本研究の主な貢献は次のとおりである。

$\bullet$ For the first time, we propose a system-hardware co-design solution Stratum for MoE serving that leverages Monolithic 3D-Stackable DRAM. Our approach heterogeneously integrates high-density Mono3D DRAM dies with high-performance logic dies via 3D hybrid bonding, and further integrates this Mono3D DRAM stack with GPUs using a 2.5D silicon interposer. This architecture serves as a high-throughput and cost-effective alternative to conventional GPU-HBM-based MoE serving systems.

$\bullet$ At the hardware level, we introduce an in-memory tiering mechanism that exploits the inherent access latency variations across Mono3D DRAM layers resulting from vertical scaling. Additionally, we propose an NMP processor tailored for hybrid-bonding-based Mono3D DRAM, incorporating optimized data mapping and communication strategies for both expert and attention execution.

$\bullet$ At the system level, we observe the nonuniform activation frequency of experts depending on user request topics. Based on this, we classify experts into hot and cold categories and assign them to fast and slow tiers of Mono3D DRAM, respectively. The proposed topic-aware serving system queues and dispatches requests according to their topics, predicted by our and lightweight topic classifier, while adhering to defined service-level objectives (SLOs).

$\bullet$ Cross-layer evaluations (device, circuit, algorithm, and system) demonstrate that Stratum achieves up to $8.29\times$ better decoding throughput and $7.66\times$ better energy efficiency in practical MoE serving scenarios, compared to state-of-the-art GPU-baselines.

<span id="section-2"></span>

## 2 背景

<span id="section-2-1"></span>

### 2.1 モノリシック 3D スタッカブル DRAM

Mono3D DRAM は DRAM スケーリングを継続する有望な技術であり、学術界と産業界の双方から大きな注目を集めている [Har21, Tra22, Dri17, Att24, Hig21]。従来の 2D DRAM 技術と比べ、ナノシート電界効果トランジスタ (FET) による厳密なゲート制御と積層チャネル構造、3D NAND Flash に着想を得た層ごとの成膜や高アスペクト比エッチングによる超薄型誘電体分離、密な垂直統合などの技術で垂直スケーリングを実現し、大幅に高いメモリ密度を提供する [Ong23, Sig25, A24, Mon25]。

Mono3D DRAM はモノリシック 3D スタッカブル水平 1T1C DRAM セルを用い、[図 2](#figure-02) に示すように、ワード線 (WL) の階段構造と垂直接続されたビット線 (BL) で複数層のメモリセルを接続する。HBM は TSV 製造の歩留まりの低さとダイ積層に必要な複雑なパッケージングにより高コストになる。一方、Mono3D DRAM は TSV を使わず、同じウェハ上に追加の DRAM 層を順に形成するモノリシック 3D 統合によってスケーラビリティを高め、コスト上の利点を得る。さらに、薄いダイとモノリシック統合による垂直熱伝導の改善により、熱面でも利点がある。

コストと熱の利点に加え、Mono3D DRAM はロジック層へ高いメモリ帯域も提供する。異種統合 [A24, Mon25] を利用し、メモリセルとロジック周辺回路間の高速データ転送に Cu–Cu ハイブリッドボンディングを用いる。[図 3](#figure-03) は同じ 2.5D 統合プラットフォーム上で Mono3D DRAM と HBM を比較する。HBM の内部帯域は、10 $\mu$m という粗いピッチの TSV [Sma16] に制約されるため、帯域が限られ、面積オーバーヘッドがメモリ密度を低下させる。これに対して Mono3D DRAM は、1 $\mu$m [Che20a] というはるかに微細なピッチで DRAM ダイとロジックベースダイを Cu–Cu ハイブリッドボンディングし、BEOL 金属配線で接続することで、極めて高い内部帯域を実現する。

内部帯域が高いにもかかわらず、[図 3](#figure-03) に示すように、Mono3D DRAM はインターポーザ I/O インターフェースの帯域が限られるため、HBM と同様に外部帯域の制約を受ける。さらに、先行研究 [Fin17] は、ロジックベースダイ上の配線とインターポーザ I/O インターフェースを通る外部プロセッサへのデータ転送が大きなエネルギーを消費することを示している。これらの非効率性から、内部帯域を活用してエネルギー効率を高めるには、Mono3D DRAM とともにロジックダイへ NMP を統合する必要がある。

Mono3D DRAM は非常に大きなメモリ容量を期待できる一方、層ごとのアクセス遅延の大きな差が垂直スケーリングを制限する。[図 2](#figure-02) に示すように、階段構造の下部にある WL ほど、線形に伸びる WL 配線による寄生容量と抵抗が増加する。この遅延の不均衡は、Mono3D DRAM を数百層へ拡張すると顕著になる。最悪遅延に合わせて設計するのではなく、この遅延の異質性を利用することでシステムレベルの性能を改善できる。この課題は、[第 3 節](#section-3) で詳述する *in-memory tiering* というアーキテクチャを自然に導く。Mono3D DRAM のスケーリング傾向は 3D NAND Flash と整合する。Mono3D DRAM は 400 層を超えてスケールした同様の製造工程を利用するためである [A25]。さらに、近年のホワイトペーパーは 500 層、さらには 1000 層まで拡張できる可能性を示している [New25, Sca24a]。これらの進展と垂直スケーリングの見通しを踏まえ、近い将来の実現可能性を反映して最大 1024 本のワード線 (WL) スタックを仮定する。

<span id="figure-02"></span>

![図 2. Monolithic 3D-Stackable DRAM with vertically stacked horizontal 1T1C DRAM cells. Bitlines are vertically routed to avoid sense margin variations, and wordlines are routed through staircases. The activation latency varies by layers due to wordline staircases.](../../papers/stratum/figure-02.png)

**図 2.** Monolithic 3D-Stackable DRAM with vertically stacked horizontal 1T1C DRAM cells. Bitlines are vertically routed to avoid sense margin variations, and wordlines are routed through staircases. The activation latency varies by layers due to wordline staircases.

<span id="figure-03"></span>

![図 3. HBM versus Mono3D DRAM on 2.5D integration platform with a xPU die. The HBM and Mono3D DRAM are attached to the logic base die through TSVs and Cu-Cu hybrid bonding, respectively.](../../papers/stratum/figure-03.png)

**図 3.** HBM versus Mono3D DRAM on 2.5D integration platform with a xPU die. The HBM and Mono3D DRAM are attached to the logic base die through TSVs and Cu-Cu hybrid bonding, respectively.

<span id="section-2-2"></span>

### 2.2 Mixture of Experts LLM

LLM のスケーリング則 [Kap20] が示すように、密な Transformer モデルは大きくなるほど精度が向上するが、学習とサービスのコストも増加する。OLMoE [Olm24]、Mixtral [Jia24a]、Deepseek V3 [Dee24a]、Time MoE [Tim24]、DBRX [Dbr25]、LLaMA-4 [Lla25]、Kimi-K2 [Kim25e] などの近年の MoE モデルは、各 token で少数のエキスパートだけを活性化することで魅力的な代替となる。この疎な活性化は学習のスケーラビリティを高め、事前学習コストを比例して増やさずに大きなパラメータ数を扱える [Sca24] 一方、推論コストをより小さな密なモデルに近く保つ [Fed22]。他方、MoE モデルにはルーティング機構が必要であり、ゲーティングネットワークが学習済みルーターパラメータを使い、token 表現 (FFN 入力または中間活性) からエキスパート割当スコアを計算して疎な選択パターンを決める [Fed22]。各 token は選択されたエキスパートへ送られて個別に処理され、複数のエキスパートを使う場合は通常、ルーティングスコアによる重み付き集約で出力を結合し、層の最終出力を生成する [Jia24a, Dee24a, Fed22]。

MoE の MLP モジュールが切り替わる性質は、ハードウェア展開に固有の課題をもたらす。第一に、MoE モデルは大きく、エキスパート重みが全体の大部分を占める。たとえば Mixtral $8 \times 7$B ではモデルの 95% 以上がエキスパート重みであり [Jia24a]、GPU メモリに大きな負荷をかける。第二に、token ごとのエキスパート使用状況は動的に変化し事前には分からないため、異なる計算ユニットへエキスパートを分散すると負荷不均衡が生じる [Dee24a]。近年は使用状況を事前予測して通信オーバーヘッドを削減する研究が進められている。ExpertFlow [Exp24a] は軽量な代理モデルでルーティング経路を予測し、MoE Infinity [Moe25] は層をまたぐ活性プロファイリングから統計的にエキスパート選択を予測する。GPU と近メモリ処理を組み合わせたシステムでは、Duplex [Dup24] が遅延モデルと batch size に基づき、エキスパート計算を GPU または NMP ユニットへ動的に割り当てる。

学習中の MoE モデルには通常、エキスパートの飢餓状態 (一部のエキスパートだけが極端に少なく選ばれる状態) を防ぎ、より均一な利用を促すエキスパート不均衡損失が含まれる [Fed22, Jia24a]。しかし学習が進むと、エキスパート間にドメイン特化が自然に現れる [Acc23, Chi19, Exp24b]。エキスパート数が増え、共有エキスパートが導入されると、この特化はさらに強くなり、共通知識を集約しつつルーティングされたエキスパートのドメイン固有性を高める [Olm24, Dee24a, Dai24, Lla25]。この観察に基づき、GPU のみの環境で特定ドメインへのエキスパート親和性を利用し、推論を高速化する研究が行われている [Exp24b, Apt24, Moe25a]。

<span id="figure-04"></span>

![図 4. Expert hit profiling from LLaMA-4 Scout (16 Experts).](../../papers/stratum/figure-04.png)

**図 4.** Expert hit profiling from LLaMA-4 Scout (16 Experts).

プロファイリングの結果、エキスパートの使用状況はクエリのトピックと明確に関係し、特定のトピックでは特定のエキスパートが大幅に高頻度で活性化することを観測した。[図 4](#figure-04) の例では、LLaMA-4 Scout が MMLU サブセットの数学・論理トピックで 90% を超えるドメイン固有のエキスパート親和性を示す。サービス時には、まずオフラインプロファイリングで各トピックのエキスパート命中率 (使用確率) を収集し、オンラインではスケジューラ内の軽量トピック分類器が batch の全クエリにトピックラベルを付与する。この分類に基づき、システムは頻繁に使われるエキスパートを高速な Mono3D DRAM 層へ配置してアクセス遅延を最適化する ([第 5 節](#section-5))。

<span id="section-3"></span>

## 3 Stratum の概要

<span id="section-3-1"></span>

### 3.1 システム概要

<span id="figure-05"></span>

![図 5. Example Stratum configurations.](../../papers/stratum/figure-05.png)

**図 5.** Example Stratum configurations.

<span id="figure-06"></span>

![図 6. Serving system based on Stratum.](../../papers/stratum/figure-06.png)

**図 6.** Serving system based on Stratum.

<span id="figure-07"></span>

![図 7. Stratum NMP architecture. (a) Overview of the processor at the chip level. Microarchitectures of (b) the processing unit (PU) at the channel level, and (c) the processing element (PE) at the bank level.](../../papers/stratum/figure-07.png)

**図 7.** Stratum NMP architecture. (a) Overview of the processor at the chip level. Microarchitectures of (b) the processing unit (PU) at the channel level, and (c) the processing element (PE) at the bank level.

Stratum の処理システムは、シリコンインターポーザで接続された xPU ダイと、数を構成可能な Monolithic 3D-Stackable DRAM チップからなり、近メモリ計算能力を備える。異なる数の Mono3D DRAM チップを使い、モデルサイズに応じた 3 種類の構成例を示す ([図 5](#figure-05))。*Stratum-L* は NVIDIA H100 計算ダイを xPU とし、6 個の Mono3D DRAM チップをインターポーザで接続する。*Stratum-S* は NVIDIA RTX A6000 ダイと 32 GB の単一 Mono3D DRAM チップを使う。*Stratum-XL* は *Stratum-L* モジュール 2 個で構成され、より大きなモデルのサービス向けに合計 384 GB のメモリを提供する。これらの構成は多様な計算・メモリ要件に対応し、NVLink などのチップ間インターコネクトで拡張できる [The22]。

各 Mono3D DRAM チップは上部のメモリダイと下部のロジックダイからなり、Cu-Cu ハイブリッドボンディングで接続して高い内部帯域を実現する。さらに、Mono3D DRAM の垂直層間にあるアクセス遅延差を利用するため、メモリダイ内に内部メモリ階層化を導入する。下部ロジックダイには強力な近メモリプロセッサ (NMP) を実装し、常にホストプロセッサへデータを取り出すことなく LLM 推論を支える ([第 3.2 節](#section-3-2))。

[図 6](#figure-06) describes the flow of a serving system based on Stratum. In a realistic serving scenario, queries submitted by users are of varying topics. When users send inference requests, the host processor uses a lightweight topic classifier to determine the topic of the query. These requests are then enqueued in the serving queue with a topic tag. Periodically, the scheduler groups inference requests from the serving queue and later dispatches them to the Stratum processing system. To enhance user experience, a key Service-Level Objective (SLO) is Time to First Token (TTFT), which ensures that a request does not wait too long before processing begins. When SLO permits, the scheduler prioritizes batching requests of the same topic to maximize the benefits of expert placements. The memory mapper constructs the aggregated expert hit prediction for the batch by consulting the pre-profiled expert usage table and produces a target placement as a mapping between experts to Mono3D DRAM layers. Expert swaps are executed before every new batch with different topic tags to meet the target layout. Considering the arithmetic intensity of each stage, the Computation Mapper assigns the prefill phase to xPU and the decode phase to the Stratum NMP, following a similar strategy as in [Att24]. Additionally, the lightweight topic classification is executed by the host processor.

<span id="section-3-2"></span>

### 3.2 Stratum の近メモリ処理

[図 7](#figure-07) は Stratum NMP のアーキテクチャを示す。3D 統合の利点を活用するため、処理コンポーネントをチップ、channel、bank というメモリ階層の複数レベルに配置する。この構成は、MoE モデルの主要なボトルネックである注意計算とエキスパート計算の高速化を目的とする。

[図 7](#figure-07)(a) はロジックダイプロセッサと Mono3D DRAM ダイの統合を示す。ロジックダイには複数の処理ユニット (PU) があり、それぞれ専用の Mono3D DRAM channel と結合する。PU は双方向リング型のチップ内ネットワークで接続され、reduce-scatter や all-gather など LLM ワークロードのデータ通信を最適化する。このリングネットワークは NMP モードでのみ使われる。通常のメモリ動作ではロジックダイの NMP は停止し、従来のメモリアクセスへの干渉を最小限にする。NMP モードでは xPU が標準 DRAM インターフェースを通じて、クエリや隠れ token ベクトルなどの入力を Mono3D DRAM bank の予約行へストリームする。計算完了後、xPU は専用アドレス空間へアクセスして結果を取得する。

各 PU は対応する DRAM channel に割り当てられたデータを処理し、Mono3D DRAM とロジックダイ間の垂直配線量が膨大であることから重要な cross-channel アクセスを避ける。[図 7](#figure-07)(b) に示す PU マイクロアーキテクチャは、near-bank 処理要素 (PE) クラスタ、共有メモリ、特殊関数エンジン、リングルータ、リデューサからなる。near-bank PE クラスタは GeMM と GeMV の双方に最適化された複数の PE を統合する。並列リダクションツリーで実装した channel 内リデューサは、必要に応じて channel 内の複数 PE の部分和 (psum) を集約する。リングルータは PU 間通信で効率的にデータをルーティングするローカルスイッチと、その場でデータを削減するアグリゲータを備える。入力データストリームは共有メモリを経由せず、ルータで直ちに累積できる。累積結果は PU 内に保存するか、必要に応じて隣接 PU へ転送する。特殊関数エンジンは注意機構の `Softmax` や、エキスパート層の `SiLU`、`GeLU` などの活性化関数を実行する。ベクトルレジスタファイル、スカラレジスタファイル、複数の算術ユニットを含み、single-instruction-multiple-data (SIMD) 方式で複雑な関数を単純なプリミティブへ分解し、レジスタファイル内でオペランドと中間結果を読み書きしてデータ再利用を最大化する。

 [図 7](#figure-07)(c) に示す bank レベルでは、各 PE が GeMM と GeMV を実行する。bank レベル PE は、行列レジスタファイル、psum メモリ、単純なローカルメモリコントローラを備えたテンソルコアからなる。対応する DRAM bank に直接接続するメモリコントローラは、プログラム可能な階層化テーブルで行アドレスを特定のメモリ層 ID に動的変換し、`tRCD` を適応制御して性能を最適化する。行交換バッファは一時的な行データを保持し、明示的な外部フェッチなしで層間データ移動を可能にする。テンソルコアは $n$ 個の並列 $k$-tap 内積エンジンと $n$ 個のローカルアキュムレータを備える。ダブルバッファ構成の psum メモリは中間結果の蓄積と出力転送を同時に支援する。処理済み出力は特殊関数エンジンへ送って要素単位の関数を評価するか、後続計算のため channel レベル共有メモリへ戻せる。

混合ボンディングによる Mono3D DRAM 統合に最適化した Stratum のアーキテクチャは、AttAcc [Att24]、Neupims [Neu24a]、Duplex [Dup24] など HBM 中心の NMP 方式とは異なる。チップ内リングネットワークは all-gather や reduce-scatter など MoE 推論の通信パターンを支え、Duplex [Dup24] の集中型グローバルバッファと crossbar を不要にするため、スケーラビリティを高め物理設計を簡素化する。専用 `Softmax` ユニットに依存する Duplex [Dup24] や AttAcc [Att24] と異なり、SIMD エンジンはプログラム命令で一般的な非線形演算子を実行する。また、プロセッサをロジックダイに完全実装して Mono3D DRAM ダイへハイブリッドボンディングすることで、AttAcc [Att24] と Neupims [Neu24a] に見られる DRAM 製造工程の制約と TSV 帯域制限を避ける。回路レベルでは、階層化テーブルと行交換バッファなど Mono3D DRAM 固有のプリミティブを導入し、階層化メモリの遅延を利用して MoE サービスのエキスパート移動を高速化する。

<span id="section-4"></span>

## 4 Stratum の演算子マッピングと実行

<span id="section-4-1"></span>

### 4.1 エキスパート処理

MoE 層の実行は、token ルーティング、エキスパート計算、結果集約の 3 段階からなる。[図 8](#figure-08)(a) に示すように、batch の token は xPU が計算したルーティング結果に基づいて異なるエキスパートへ送られる。ルーティングは通常、入力 4096・出力 8 次元などの軽量な線形層であり計算コストが小さいため、これが可能となる。その後、少なくとも 1 token が割り当てられた活性化済みエキスパートだけを実行する。最後に、全エキスパートの出力を重み付き和で統合して最終出力 token を生成する。エキスパート計算と結果集約はいずれも Stratum NMP プロセッサが実行する。

<span id="figure-08"></span>

![図 8. (a) Example of MoE’s token-to-expert mapping. (b) The computation stages of an expert with $M$ routed tokens and matrix partition, assuming four PUs for simplicity. (c) The step-by-step execution of the MoE layer in Stratum.](../../papers/stratum/figure-08.png)

**図 8.** (a) Example of MoE’s token-to-expert mapping. (b) The computation stages of an expert with $M$ routed tokens and matrix partition, assuming four PUs for simplicity. (c) The step-by-step execution of the MoE layer in Stratum.

MoE モデルの 1 エキスパートの計算は通常、[図 8](#figure-08)(b) に示すように 3 つの GeMM を連鎖させる [Jia24a, Lla25]。現在の batch で 1 エキスパートへルーティングされた token 数を $M$、隠れ次元を $K$、中間次元を $N$ とする。まず、サイズ $M\times K$ の入力隠れ行列 $\mathbf{X_{1}}$ に、サイズ $K\times N$ の 2 つの重み行列を乗じ、サイズ $M\times N$ の中間行列 $\mathbf{Z_{1}}$ と $\mathbf{Z_{2}}$ を得る。$\mathbf{Z_{1}}$ に非線形な要素単位活性化を適用し、その結果と $\mathbf{Z_{2}}$ のアダマール積から $\mathbf{X_{2}}$ を作る。最後に、$\mathbf{X_{2}}$ にサイズ $N\times K$ の射影ダウン重み行列を乗じ、サイズ $M\times K$ の出力 $\mathbf{Z_{3}}$ を得る。

**分割戦略。** 実際には、エキスパートごとに受け取る token 数が異なる。また、エキスパートは Mono3D DRAM 階層内の異なる層に配置され、メモリアクセス遅延も異なるため、負荷不均衡がさらに悪化する。複数のエキスパートを PU に分散すると PU 間で深刻な負荷不均衡が生じうるため、選択されたエキスパートは 1 つずつ順番に実行する。すべての PU がテンソル並列で 1 エキスパートの処理に協調するには、3 つの GeMM に関わる各行列を tile に分割し、並列実行のため PU に割り当てる必要がある。[図 8](#figure-08)(b) は、簡単のため 4 PU を仮定した Stratum の行列分割を示す。異なる次元で分割すると、入力複製、重み複製、部分和集約のトレードオフが生じる。メモリ使用量の大部分を占めるエキスパート重みの複製を避けるため $M$ 次元では分割しない。代わりに GeMM1 と GeMM2 の重み行列を垂直に、GeMM3 を水平に分割する。この方法は、最初に ${\bf{X}}_{t}$ を複数 PU に複製し、${\bf{Z}}_{3}$ の部分結果を複数 PU から集める代わりに、上向き射影と下向き射影の間のデータ通信をなくす。すべての活性化エキスパートの入力行列 ${\bf{X}}_{1}$ は batch 内 token の集合である ${\bf{X}}_{t}$ から得られるため、${\bf{X}}_{t}$ の複製コストは十分に償却される。さらに、複数 PU からの ${\bf{Z}}_{3}$ の収集と削減は次のエキスパート処理と並列に実行でき、遅延を隠蔽できる。

**実行段階。** [図 8](#figure-08)(c) は MoE 層の逐次実行を示す。xPU は batch の入力 token、対応するエキスパート ID、スケーリング重みを Mono3D DRAM へ送り、NMP モードへ切り替える (step 1)。採用した行列分割により、各 Mono3D DRAM channel は入力 token 行列全体を受け取る必要がある。次に Stratum NMP プロセッサは step 2–7 で活性化エキスパートを順に実行する。step 2 と 3 では、全 PE のテンソルコアが 2 つの上向き射影 GeMM を実行して中間結果 $\mathbf{Z}_{1}$、$\mathbf{Z}_{2}$ を計算する。step 4 と 5 では特殊関数エンジンが活性化関数とアダマール積を計算する。行列分割により、第 3 GeMM に必要な入力 slice を得るための PU 間通信は不要である。step 6 で第 3 GeMM を実行し、reduce-scatter で PU 間の最終出力行列 $\mathbf{Z}_{3}$ を集約する。残りの活性化エキスパートについて step 2–7 を繰り返す。step 9 では特殊関数エンジンがエキスパート出力の重み付き和を計算して最終出力 token を作り、指定した DRAM アドレスへ書き戻す。最後に step 10 で Mono3D DRAM は NMP モードを終了し、xPU が指定アドレス空間から計算済み token を取得する。

<span id="figure-09"></span>

![図 9. Optimized timing diagram of the expert processing.](../../papers/stratum/figure-09.png)

**図 9.** Optimized timing diagram of the expert processing.

**実行最適化。** [図 9](#figure-09) は計算資源と通信資源の利用率を最大化する最適化パイプラインを示す。まず xPU から Mono3D DRAM への転送準備遅延を減らすため、入力 token 行列を複数 slice に分割し、各 slice を異なる Mono3D DRAM channel へ送る。これで入力準備のオーバーヘッドを減らし、高速なロジックダイリングネットワークによる後続の all-gather で全 PU 用の完全な入力行列を再構成する。次に GeMM2 と活性化関数評価にはデータ依存がないため重ねて実行し、パイプライン利用率を高める。第三に、GeMM3 の reduce-scatter 通信を次のエキスパートの GeMM1 と並列化し、通信遅延を計算に隠す。最後に、エキスパート出力が利用可能になると直ちに特殊関数エンジンが重み付き和を実行し、アイドルサイクルを減らして全体のスループットを高める。

高帯域の共有メモリにより、各 PU 内の PE 間通信オーバーヘッドは無視できる。したがって PU 内の行列分割は、主にテンソルコアのマッピング利用率を最大化することに集中する。重み行列の長い次元を分割し、得られたサブタイルを PE に配って並列処理する。そのため、上向き射影の重み slice ${\mathbf{W}}_{1,2}[i]$ は通常水平に分割し、下向き射影の重み slice ${\mathbf{W}}_{3}[i]$ は計算効率を高めるため PE 間で垂直に分割する。

<span id="section-4-2"></span>

### 4.2 アテンション処理

大規模言語モデル (LLM) の生成タスクは、key–value (KV) キャッシュへのデータアクセスがボトルネックになることが多い。Stratum は Mono3D DRAM とベースダイ上の NMP ロジック間の高帯域を利用してこの問題に対処する。ただし帯域を十分に活用するには、DRAM 層から垂直に取得したデータを適時に処理することが重要である。そうしなければ、ロジックダイ内の計算または通信ボトルネックによって利用可能な帯域が使い切れない。

<span id="figure-10"></span>

![図 10. Execution of attention layer. (a) Heads (e.g., eight) assignment across PU groups (e.g., four). Intra-PU group: (b) Attention operator mapping. (c) Concurrent processing of multiple heads (e.g., two).](../../papers/stratum/figure-10.png)

**図 10.** Execution of attention layer. (a) Heads (e.g., eight) assignment across PU groups (e.g., four). Intra-PU group: (b) Attention operator mapping. (c) Concurrent processing of multiple heads (e.g., two).

注意 head 間にデータ依存がないため、Stratum は head レベル並列性を利用して注意計算を効率よく実行する。[図 10](#figure-10)(a) はロジックダイ上の注意 head タスク割当を示す。1 グループのリクエストに含まれる複数の head を Mono3D DRAM デバイスへ割り当てられる。割当 head 数はネットワークモデルによって変わり、MoE モデルで一般的な grouped query attention [Lla25, Jia24a] や、サービス遅延要件下のリクエスト並列度に依存する。多様な head 並列性に対応するため、ロジックダイの PU を異なるサイズの複数 PU グループへ柔軟に分割する。ただし各グループ内の PU は、[図 10](#figure-10)(a) のようにチップ内リングトポロジで隣接接続されていなければならない。矢印でつながれた PU がリング上の PU を表す。この構成は高速な双方向リンクによる効率的なグループ内通信も可能にする。スループットとハードウェア利用率を高めるため、各グループには少なくとも 2 head を割り当て、異なる計算段階をインターリーブする。例えば、一方の head が線形演算を行う間に、もう一方が `Softmax` を実行できる。

[図 10](#figure-10)(b) は 1 head の key 行列と value 行列を PU グループ内の PU に分割する方法を示す。通常、系列長次元 (512–32k token など) は注意 head 次元 (64–128 など) より大幅に大きいため、系列長次元で分割する。しかし `Softmax` は全 token にわたるグローバル情報、すなわち正規化に使うグローバル最大値 ($\mathrm{row}_{\max}(\mathrm{Scores})$) と指数のグローバル和 ($\sum\exp(\mathrm{Scores}-\mathrm{row}_{\max}(\mathrm{Scores}))$) を必要とする [A20]。各 PU は専用特殊関数エンジンで局所最大値と和を独立計算でき、グローバル値の導出に必要なのは PU 間のスカラー交換だけである。デコード段階で PU の負荷を均衡させるため、新たに生成された key-value ペアを PU グループ内の PU へラウンドロビン方式で分配する。

[図 10](#figure-10)(c) は PU グループ内の複数 head の最適化実行フローを示す。まず xPU が計算済み key-value ペアを対応する DRAM channel に書き込む。クエリ (grouped query 行列の場合もある) を slice に分割し、各 slice を PU グループ内の異なる DRAM channel に割り当てる。続いて、グループ内の全 PU が MoE 層と同様のサブリング all-gather で完全なクエリ行列を得る。同じ PU グループに複数 head を割り当てる場合、`Softmax` を $\mathrm{query}\times\mathrm{key}$ と $\mathrm{attn.}\times\mathrm{value}$ の演算にインターリーブして全体遅延を最小化できる。[図 10](#figure-10) に示すように、`Softmax` は 2 回の PU 間通信を含む 3 段階に分割する。最後に、1 番目の head の reduce-scatter 遅延を 2 番目の head の $\mathrm{attn.}\times\mathrm{value}$ 演算に隠す。

まとめると、Stratum はデータ配置、演算子マッピング、スケジューリングを最適化し、ハイブリッドボンディングが可能にする垂直帯域を最大限に利用する。エキスパート計算には全 PU にわたるテンソル並列性を、注意計算にはグループ化 PU の head 並列性を適用する。両戦略はハイブリッドボンディング I/O を通じて大半のメモリアクセスをローカル Mono3D DRAM bank へ向ける。all-gather、reduce-scatter、スカラー交換など残りの PU 間通信はチップ内リングネットワークが効率よく支える。さらに、スケジューラは GeMM や GeMV などの行列演算と `SiLU`、`Softmax` などの特殊関数計算をオーバーラップさせ、チップ内通信と計算を協調させて全体の並列性を高める。

<span id="section-4-3"></span>

### 4.3 物理制約下の設計

ハイブリッドボンディングで Mono3D DRAM とロジックダイプロセッサを統合するには、熱制約と面積制約の双方を満たす必要がある。NMP モードでは、熱解析で決まるピーク電力予算 $P_{\mathrm{peak}}$ によって制限される可能性があり ([第 6.2.2 節](#section-6-2-2))、電力制約は次のようになる。

<span id="equation-01"></span>

$$
\begin{array}[]{l}{P_{\mathrm{dram}}}+{P_{\mathrm{compute}}}+{P_{\mathrm{misc}}}\leq{P_{\mathrm{peak}}},\\
{P_{\mathrm{dram}}}=\mathrm{BW}_{\mathrm{fast\_tier}}\cdot{E_{b}},\;\;\,{P_{\mathrm{compute}}}={N_{mac}}\cdot{f_{\mathrm{logic}}}\cdot{E_{\mathrm{mac}}}.\end{array}
$$

ここで $\mathrm{BW}_{\mathrm{fast\_tier}}$ は Mono3D DRAM の最速層のピーク帯域、$E_{b}$ はハイブリッドボンディングを介して DRAM 層からロジックダイへデータを転送する 1 bit あたりのエネルギー、$N_{mac}$ はテンソルコアの積和 (MAC) ユニット総数、$f_{\mathrm{logic}}$ はロジックダイの動作周波数、$E_{\mathrm{mac}}$ は 1 MAC 操作あたりのエネルギーである。雑多電力 $P_{\mathrm{misc}}$ には、ロジックダイ SRAM、レジスタファイル、ルータ、特殊関数エンジン、PU 内リデューサ、ローカルメモリコントローラが含まれ、演算子の種類とデータフローに応じて変化する。

ハイブリッドボンディングによるデータ I/O はロジックダイのアクティブ面積を消費しないが、DRAM とロジックの両ダイへ電力を供給する TSV は必要である [Exp24]。したがって、次の面積制約を満たさなければならない。

<span id="equation-02"></span>

$$
A_{\mathrm{PD}}+N_{mac}\cdot A_{\mathrm{mac}}+A_{\mathrm{PHY}}+A_{\mathrm{peri}}+A_{\mathrm{misc}}\leq\alpha A_{\mathrm{chip}},
$$

$A_{\mathrm{PD}}$ は電力供給用 TSV の総面積、$A_{\mathrm{mac}}$ は $f_{\mathrm{logic}}$ で動作する 1 MAC ユニットの面積、$A_{\mathrm{PHY}}$ は xPU-DRAM インターフェースの物理通信層の面積、$A_{\mathrm{peri}}$ は D/Q バッファやレベルシフタなどロジックダイ上の低電圧 Mono3D DRAM 周辺回路の面積、$A_{\mathrm{misc}}$ は $P_{\mathrm{misc}}$ と同様のその他ロジック面積、$\alpha$ は目標利用率を表す。面積 $A_{\mathrm{TSV}}$ の 1 本の TSV が電流 $I_{\mathrm{TSV}}$ を供給できるとすると、TSV の総面積は次で与えられる。

<span id="equation-03"></span>

$$
\begin{array}[]{l}{A_{\mathrm{PD}}}=(\frac{{{P_{\mathrm{dram\_c}}}}}{{{V_{\mathrm{dram\_c}}}}}+\frac{{{P_{\mathrm{dram\_p}}}}}{{{V_{\mathrm{dram\_p}}}}}+\frac{{{P_{\mathrm{compute}}}+{P_{\mathrm{misc}}}}}{{{V_{\mathrm{logic}}}}})\frac{{{A_{\mathrm{TSV}}}}}{{{I_{\mathrm{TSV}}}}},\\
{P_{\mathrm{dram\_c}}}+{P_{\mathrm{dram\_p}}}={P_{\mathrm{dram}}}\end{array}
$$

$V_{\mathrm{dram\_c}}$、$V_{\mathrm{dram\_p}}$、$V_{\mathrm{logic}}$ はそれぞれ Mono3D DRAM コア、高電圧周辺回路、低電圧ロジックダイの電源電圧を表す。式 ([1](#equation-01))、([2](#equation-02))、([3](#equation-03)) をロジックダイプロセッサの構成設計の指針として用いる ([第 6.2.3 節](#section-6-2-3))。

<span id="section-5"></span>

## 5 Stratum のアルゴリズム-システム協調最適化

<span id="section-5-1"></span>

### 5.1 エキスパート使用量の予測

 [第 2.2 節](#section-2-2) で述べたように、事前学習済み MoE モデルは推論時にドメイン固有のエキスパート特化を示すことが多い [Exp24b] ([図 4](#figure-04))。MoE 推論の主要課題の一つは全エキスパートの巨大な総パラメータを扱うことなので、この特化は効率的な推論とサービスの好機となる。エキスパート特化が特定のクエリトピックと一致すれば、MoE エキスパートの配置を最適化できる。あるトピックで使用確率 (命中率) が高いエキスパートを高速な Mono3D DRAM 層へマッピングすれば、DRAM からロジックベースダイへのデータ転送遅延を削減できる。

MoE エキスパートのマッピングを可能にする Stratum の重要な要素が、入力クエリにタグを付けるトピック分類器である。これにより Stratum スケジューラは各クエリのトピック分布を推定できる。トピックごとのエキスパート使用表 ([図 6](#figure-06)) と組み合わせ、スケジューラはエキスパートの重み行列を適切な層へ割り当てる。実装では、Stratum に基づくオンラインサービスシステムの一部として、6 トピック・67M パラメータの DistillBERT ベース [Ber19, San19] 分類器を学習した。標準 NLP データセットから実サービスの多様なプロンプト様式への分布シフトに対応するため、GPT-4o による書き換えを用いたデータ合成で学習データを拡張した。分類器は小型であり、実験環境の中程度のリクエストレート (毎秒 4 クエリ未満) ではデコード 1 step あたりの遅延オーバーヘッドが 2% 未満で、実サービスデータセット (Chatbot Arena 会話 [Chi24]) の 6 トピックモデルでそれぞれ 85.0% と 81.0% の分類精度を達成した。データ拡張、学習、評価の詳細は [第 6.3.1 節](#section-6-3-1) に示す。

<span id="section-5-2"></span>

### 5.2 データ配置戦略

<span id="figure-11"></span>

![図 11. Example expert placement optimization for Mono3D DRAM-NMP system with tiered memory.](../../papers/stratum/figure-11.png)

**図 11.** Example expert placement optimization for Mono3D DRAM-NMP system with tiered memory.

<span id="algorithm-01"></span>

**Algorithm 1. Expert Weight Placement.**

```text
Require: number of layers L; experts per layer K; active experts k;
        usage frequencies F = {f_p^l}; expert size S_E (bytes);
        DRAM banks N_bank; row-buffer size S_rb (bytes);
        rows reserved for NMP data Phi.
Ensure: DRAM row-address intervals [a_p^l, b_p^l] for every expert.

Delta <- ceil(S_E / (N_bank * S_rb))       // rows occupied by one expert
tau <- kL                                      // threshold of fast experts
Sort F in descending order as <f_(p_1)^(l_1), ..., f_(p_KL)^(l_KL)>
for i <- 1 to KL do
  if i <= tau then
    a_(p_i)^(l_i) <- (i - 1) * Delta
  else
    a_(p_i)^(l_i) <- Phi - (KL - i + 1) * Delta
  end if
  b_(p_i)^(l_i) <- a_(p_i)^(l_i) + Delta - 1
end for
return {[a_p^l, b_p^l] | p in [1, K], l in [1, L]}
```

Stratum は MoE モデル内のデータを、ホットエキスパート重み、コールドエキスパート重み、KV キャッシュ、非 NMP データの 4 種類に分類する。ホットエキスパートには共有エキスパートと、特定トピックでルーティング命中確率が高いその他のエキスパートが含まれる。非 NMP データは位置埋め込み、レイヤーノルムのシフト・スケールなどの雑多なパラメータで、通常は NMP ではなく外部プロセッサが計算に使う。メモリ層ごとの異なるアクセス遅延を利用することで、サービス性能を高めるデータ配置を最適化できる。

 [図 11](#figure-11) に示すように、Stratum は xPU が処理する非 NMP データを最も遅いメモリ層に置く。アクセス時にインターポーザのボトルネックを通る必要があり、最遅層の内部 DRAM 帯域より 1 桁遅いためである。これにより高速な層を NMP ワークロード専用に保てる。Stratum はトピック固有リクエストのオフラインプロファイリングに基づいてエキスパートをホットとコールドに分類し、ホットを高速層、コールドを低速層に割り当てる。この配置でホットエキスパートは高速 Mono3D DRAM 層の低遅延アクセスを受けられる。エキスパート重みの配置は Algorithm 1 に詳述する。各重みを shard に分割し、テンソル並列戦略に従って Mono3D DRAM bank に分散する ([第 4.1 節](#section-4-1))。Algorithm 1 で得た物理行アドレスから論理メモリ層へのマッピングは量子化に相当し、階層化テーブルで設定する ([第 3.2 節](#section-3-2))。評価では各層に同数の行を割り当てる均一マッピングを採用した ([第 6.2.1 節](#section-6-2-1))。リクエスト生成に伴い容量が動的に変わる KV キャッシュは中速メモリに置く。あるトピック (例 A) の処理が終わると、Stratum スケジューラは新しいトピック (例 B) へ移り、新トピックのエキスパート活性頻度に基づいて交換を開始する。高コストなホストプロセッサ転送を避けるため、交換は [第 3.2 節](#section-3-2) の近メモリ操作で実行する。具体的にはローカルメモリコントローラが 2 本の DRAM 行を専用行交換バッファへ一時保存し ([図 7](#figure-07)(c))、新しい行アドレスへ書き戻す。

<span id="section-6"></span>

## 6 評価

<span id="section-6-1"></span>

### 6.1 実験設定

<span id="section-6-1-1"></span>

#### 6.1.1 モノリシック 3D スタッカブル DRAM の構成

<span id="figure-12"></span>

![図 12. Mono3D DRAM bank configuration. The performance is simulated from NeuroSim Neu24b and Coventor process simulator Cov24.](../../papers/stratum/figure-12.png)

**図 12.** Mono3D DRAM bank configuration. The performance is simulated from NeuroSim [Neu24b] and Coventor process simulator [Cov24].

For Mono3D DRAM technology, we adopt the vertical bitline connections for 3D stackable horizontal 1T1C. We design the Mono3D DRAM scaled to 1024 layers and define the bank structure as in [図 12](#figure-12), where 1024 BLs $\times$ 1024 WLs form a MAT and 1024 MATs form a bank. To illustrate the impact of heterogeneous integration, [図 13](#figure-13) presents a 3D view of the proposed Mono3D DRAM bank. The high-voltage circuits are implemented beneath the memory array using a mature CMOS-under-array process, while the low-voltage circuits are fabricated on an advanced CMOS die and later hybrid-bonded to the memory tiers through Cu–Cu bonding pads. In this work, we leverage the 32 nm technology node for the CUA process and the 7 nm technology node for the bonded CMOS tier. To obtain the bank-level results, we utilize the Coventor process model [Cov24] for RC parameter extraction of the 3D DRAM array, and combine it with the peripheral circuit results extracted from NeuroSim [Neu24b] merging with the timing of DDR5 Standards [Ddr20], as shown in [図 12](#figure-12). The 1T1C model of Mono3D DRAM is built by the Coventor SEMulator3D process simulator [Cov24] based on a 3D DRAM structure specification in [Ong23]. The detailed parameters are listed in [Table 1](#table-01). The overall Mono3D DRAM achieves a memory density of 2.156 Gb/mm<sup>2</sup>, which is $5.2\times$ higher than that of the latest 32Gb DDR5 die (0.417 Gb/mm<sup>2</sup>[A24a]). It provides an internal bandwidth ranging from 19.01 TB/s to 30.34 TB/s, depending on the memory tier.

<span id="figure-13"></span>

![図 13. Mono3D DRAM array with heterogeneous integration, hybrid-bonding and CMOS-under-array (CUA).](../../papers/stratum/figure-13.png)

**図 13.** Mono3D DRAM array with heterogeneous integration, hybrid-bonding and CMOS-under-array (CUA).

<span id="table-01"></span>

![Table 1. Monolithic 3D-Stackable DRAM Parameters](../../papers/stratum/table-01.png)

**表 1.** Monolithic 3D-Stackable DRAM Parameters

<span id="section-6-1-2"></span>

#### 6.1.2 ロジックダイ・プロセッサのモデル化

Stratum ロジックダイプロセッサの各コンポーネントは SystemVerilog で実装し、7 nm 予測プロセスデザインキット ASAP7 [Asa16] と Cadence Genus [Gen24] で合成した。ハードウェアには LLM サービスで広く使われる IEEE754 FP-16 算術形式 [Iee19] を採用する。ロジックダイのローカル psum メモリと共有メモリは FinCACTI [Fin14] でモデル化した SRAM で実装し、公開 SRAM 仕様 [Coo24, A17] で校正した。Stratum NMP プロセッサの面積は合成レポートから得た。エネルギー消費は、ランダム刺激入力から注釈付けしたスイッチング活動を含む合成後ネットリストのシミュレーションで求めた。実行サイクル、チップ内通信サイクル、関連エネルギー指標は自作シミュレータから得る。シミュレータはテンソルサイズ、パラメータの層割当 (エキスパートパラメータや KV キャッシュなど)、注意 head マッピング、ルーティングされたエキスパート ID、各コンポーネントの遅延・エネルギーパラメータを入力とし、全体の実行時間とコンポーネントレベルの詳細なエネルギー分解を出力する。

<span id="section-6-1-3"></span>

#### 6.1.3 システムモデル

<span id="table-02"></span>

![Table 2. Evaluation Workload Setup](../../papers/stratum/table-02.png)

**表 2.** Evaluation Workload Setup

[表 2](#table-02) に示すモデル (MoE と通常の LLM) とシステム構成で評価する。各 GPU ベースラインと Stratum 構成は、性能を低下させず最大評価コンテキスト長を扱えるよう選んだ。GPU ベースラインは vLLM 0.8.1 [Kwo23a] のベンチマークスループットモードで評価し、Stratum の構成に応じて NVIDIA RTX A6000 または H100 SXM5 HBM3 GPU を用いる。GPU のエネルギーは NVIDIA-SMI ツールから得た。

システムレベルシミュレータは、[図 6](#figure-06) に従い、リクエスト生成器、SLO 対応スケジューラ、メモリ・計算マッパーを含み、Stratum NMP シミュレータと接続する。リクエスト生成器は、特定トピックの入力クエリが定めたレートで到着するポアソン過程をモデル化する。サービス SLO を考慮し、スケジューラは入力クエリを動的に batch 化して Stratum プロセッサへ送り、ホットエキスパート命中を最大化するため同一トピックのクエリを優先する。エキスパート使用表の事前知識を使い、メモリマッパーは batch のトピックを集約し、Algorithm 1 のようにホット命中を最大化する Mono3D DRAM 配置を計算する。ディスパッチ間でメモリを再構成してエキスパートを移動する。シミュレーション中は xPU と NMP のエネルギー・遅延を累積する。

<span id="section-6-2"></span>

### 6.2 ハードウェア評価

<span id="section-6-2-1"></span>

#### 6.2.1 3D DRAM の階層化

<span id="figure-14"></span>

![図 14. Mono3D DRAM latency across WL layers. The inset illustrates various access latencies according to the increasing WL RC delay when scaling the staircase for increasing WL layers.](../../papers/stratum/figure-14.png)

**図 14.** Mono3D DRAM latency across WL layers. The inset illustrates various access latencies according to the increasing WL RC delay when scaling the staircase for increasing WL layers.

[図 14](#figure-14) に示すように、Mono3D DRAM は異なる WL 層へアクセスする際、延長された WL 階段構造に対応してアクセス遅延がほぼ線形に増加する。WL 層を増やして垂直スケーリングすると、階段領域に対応する WL 寄生成分も増え、RC 遅延が長くなる。最下部 WL のクリティカルパスは長い遅延を受けるが、最上部 WL は短いアクセス遅延を持ち、システムレベルの最適化に利用できる。本研究では Mono3D DRAM にメモリ階層化を導入し、[図 14](#figure-14) のように層に対応する 8 つのタイミング層を定義する。高速層は最遅層より $1.6\times$ 高速にアクセスできる。

<span id="section-6-2-2"></span>

#### 6.2.2 電力と面積の予算

**Power.** The vertically integrated memory and logic dies require precise thermal modeling to determine the logic die’s power budget. We performed thermal simulations using the HotSpot [Tem03, Hot03] simulator for 3D IC. We consider high-end liquid cooling solutions with vapor chamber heat sinks. The heat sink is characterized by the following parameters: a convection capacitance of 75 J/K, a convection resistance of 0.01 W/K, and a thickness of 1 mm. The material properties include a thermal conductivity of 5000 J/(m$\cdot$K) and a specific heat capacity of $10^{6}$ J/(m${}^{3}\cdot$K). The thermal conductivity values are adopted from previous studies on vapor chamber thermal modeling [Per22a, Hea25a]. Additionally, advanced cooling fluids, such as phase change materials, achieve significantly reduced convection resistance of approximately $\mathrm{0.01\,W/K}$ [A22, Liq24]. Furthermore, we derived convection capacitance, heat sink thickness, and vapor specific heat parameters, explicitly considering the differences between conventional and vapor chamber heat sinks. Prior research demonstrates that state-of-the-art cooling methods for 3D ICs effectively manage power densities ranging up to $\mathrm{200\,W/cm^{2}}$ [The10]. Assuming full utilization of Mono3D DRAM internal bandwidth at 30.34 TB/s, each Mono3D DRAM die consumes approximately 104 W. Given the safe temperature for memory and data [Pow21], we conclude the logic die power caps at around 45W per chip.

**Area.** The Mono3D DRAM maintains compatibility with the xPU-DRAM interposer interface utilized by HBM3 [A22a], thereby requiring an HBM3 PHY module. The PHY module’s area overhead, computed for 16 physical channels each supporting 64-bit data I/O at 6.4 Gbps, totals 23.94 mm<sup>2</sup> [A24b, Sca17]. The logic die also has low-voltage Mono3D DRAM peripherals such as DQ buffer, level shifter, and address decoder, occupying 14.80 mm<sup>2</sup>. Power delivery to both Mono3D DRAM and the logic dies involves TSVs extending through the logic die from the interposer. Each TSV with an area of 25 $\mu$m<sup>2</sup> can deliver up to 36 mA [Exp24]. To accommodate peak power of 104 W for the Mono3D DRAM and 45W for the logic processor, the TSVs introduce an area overhead of 0.21 mm<sup>2</sup> when considering a 2:1 redundancy scheme. The logic die matches the Mono3D DRAM die area of 121 mm<sup>2</sup> (i.e., the base die dimensions of HBM3 [A22a]). Thus, the available area budget for the logic die processor is 82 mm<sup>2</sup>.

<span id="section-6-2-3"></span>

#### 6.2.3 ロジックダイ・プロセッサ

<span id="table-03"></span>

![Table 3. Stratum Logic Die Processor Specification](../../papers/stratum/table-03.png)

**表 3.** Stratum Logic Die Processor Specification

<span id="figure-15"></span>

![図 15. (a) Area breakdown of logic die processor; (b) Power breakdown of Mono3D DRAM-Logic Die at peak performance.](../../papers/stratum/figure-15.png)

**図 15.** (a) Area breakdown of logic die processor; (b) Power breakdown of Mono3D DRAM-Logic Die at peak performance.

[Table 3](#table-03) summarizes the specifications of the Stratum logic die processor at the PE, PU, and chip hierarchy levels. We calculated the maximum number of MAC units using [Equation 1](#equation-01), employing a simulated per-MAC-operation energy of $E_{\mathrm{mac}}=0.604$ pJ. The processor achieves a peak performance of 128 TFLOPS with 64k MAC units operating at 1 GHz. The PE tensor core is arranged into a $16 \times 16$ array, providing a balanced matrix tile size to optimize utilization across diverse GeMM sizes. Additionally, a programmable tiering table stores row addresses of the last Mono3D DRAM layer and the tRCD for each tier. The incoming row addresses are compared with eight stored addresses to expedite tRCD lookup. The communication-computation optimizations adopted enable the on-chip ring to require only 128 GB/s bandwidth per link without performance degradation based on the system-level simulation. [図 15](#figure-15) presents the area and power breakdown of the Stratum NMP stack. The total area occupied by the active logic is 76.63 mm<sup>2</sup>, which falls within the 121 mm<sup>2</sup> area budget, yielding a utilization of 63%. The area is predominantly consumed by the PEs, which dominate the PU-level area. The tiering table introduces only a minimal overhead of 0.1% of the PE area within each PE. The Stratum NMP stack reaches a peak power of 144.53 W when the fastest Mono3D DRAM tier is accessed concurrently with full tensor core utilization. The total power of the logic die is 42.67 W, including compute, on-chip communication, and logic-die memory access, under the 45W power budget.

<span id="section-6-3"></span>

### 6.3 システム評価

<span id="section-6-3-1"></span>

#### 6.3.1 アルゴリズム評価

<span id="figure-16"></span>

![図 16. Evaluation and comparison of system decoding throughput and energy efficiency.](../../papers/stratum/figure-16.png)

**図 16.** Evaluation and comparison of system decoding throughput and energy efficiency.

**Model.** Our model is based on DistilBERT [San19] with 67M parameters and designed for multi-topic text classification, supporting sequences of up to 1024 tokens. It features a compact architecture with 6 transformer layers and 12 attention heads, with a hidden dimension of 3072.

**Data.** Our model training involves a customized data mix across 6 topics. The datasets include a 2% split of Pile of Law for legal topic [Pil22], 1 out of 3 splits from atlas converse and INCLUDE for humanity topic [Atl23, Rom24], 5% split of Programming books for CS topic [Pro25], SciQ and ARC-easy for science topic [Cro17, Cla18], GSM8K and MATH for math topic [Cob21, Hen21], Atlas reasoning for logic topic [Atl25]. For the above-mentioned 6-topic configuration, the data encompasses approximately 70 million tokens.

**Training and Evaluation.** To address distribution shifts from standard NLP datasets to diverse real-world prompts, we use a GPT-4o-based data synthesis pipeline. We sample 500 prompts from the Chatbot Arena dataset [Chi24] to reflect natural user styles, then use GPT-4o with a fixed system prompt to rewrite 50% of our training data into a QA format. We use a mix of rewritten and original data to train our topic classifier on a single A100 GPU for 3 epochs of 3 hours each. For evaluation, we use the MMLU test sets [Li23e] and hand-curated 180-example subsets of Chatbot arena conversations dataset [Chi24] with the 6 topics. Our trained classifier achieves 94.5% and 85.0% accuracy on MMLU and Chatbot arena test sets, close to the performance of OpenAI O3-mini-high (96.2%, 91.1%). The inference overhead of the model is less than 10ms with ONNX runtime on a regular laptop CPU. We use OpenAI-O3 LLM-as-a-judge to classify 33,000 real-world queries from LMArena [Chi24], which shows that our six coarse-grained topics cover 93% of queries, confirming the robustness and generality of TopicBERT’s taxonomy.

<span id="section-6-3-2"></span>

#### 6.3.2 システム性能

[図 16](#figure-16) shows the normalized decoding throughput and energy efficiency when serving requests with equal input and output length. For Mono3D DRAM designs, we evaluate *no-tiering* and *tiering* approaches. In *no-tiering* design of Mono3D DRAM, Mono3D DRAM is treated as a single tier, therefore, the logic die is limited to operating under the worst memory access latency of the memory die. In *tiering*, Mono3D DRAM is divided into 8 tiers with fine-grained memory latency and data mapping optimizations given tiering. Stratum *tiering* consistently outperforms GPU baselines across all cases, averaging $8.29\times$, $5.39\times$, $6.13\times$, $4.48\times$ better decoding throughput for OLMoE, Mixtral, Qwen2.5, and Llama-4, respectively. Specifically, as decoding length grows, decoding on conventional GPUs with limited memory bandwidth becomes increasingly memory-bound, due to the quadratic complexity of the attention mechanism, explaining the growing gap of Stratum over GPU baselines. Stratum *no-tiering* as well outperforms GPU due to its higher internal bandwidth compared to HBM, even considering the worst-case latency. The internal memory tiering ([第 3.2 節](#section-3-2)) and MoE-specific data mapping optimizations ([第 5.2 節](#section-5-2)) further improve decoding throughput by averages of $1.45\times$, $1.39\times$, $1.32\times$, $1.34\times$ over *no-tiering* for the 4 models, respectively. Energy-wise, Stratum achieves up to $7.66\times$, $2.74\times$, $3.51\times$, $4.87\times$ better energy efficiency for the same decoding tasks across OLMoE, Mixtral, Qwen2.5, and Llama-4, respectively, due to cheaper memory access. We also extracted data from the previous work Duplex [Dup24] and made conservative scaling to compare with Stratum. Stratum achieves up to $2.9\times$, $2.5\times$, $3.0\times$, $2.2\times$ better throughput and $2.7\times$, $1.9\times$, $2.9\times$, $2.1\times$ energy over Duplex [Dup24] for OLMoE, Mixtral, Qwen2.5, and Llama-4.

<span id="section-6-3-3"></span>

#### 6.3.3 エキスパート配置の最適化

**Effectiveness.** To study the effectiveness of expert placement in the tiered Mono3D DRAM, we scan the hot expert hit rate for Mixtral $8 \times 7$B on Stratum-L as shown in [図 17](#figure-17). The hot expert hit rate is defined as the ratio of aggregated hot expert to total expert accesses at the token level. Across decoding lengths, accurate hot expert usage prediction brings $1.32\times$ to $1.51\times$ better throughput over a uniformly distributed expert usage, or equivalently a naively managed tiered memory. The benefit is more noticeable on smaller decoding lengths, as the MLP dominates the decoding latency more. Using our topic prediction model, we achieve 31.6%, 48.5%, and 68.9% aggregated hot expert hit rates when serving Mixtral, OLMoE, and Llama-4.

<span id="figure-17"></span>

![図 17. Impact of hot expert hit rates on (a) MLP (MoE layer) latency and (b) overall system throughput for Stratum-L.](../../papers/stratum/figure-17.png)

**図 17.** Impact of hot expert hit rates on (a) MLP (MoE layer) latency and (b) overall system throughput for Stratum-L.

<span id="table-04"></span>

![Table 4. Overhead of Expert Swap across Mono3D DRAM Tiers](../../papers/stratum/table-04.png)

**表 4.** Overhead of Expert Swap across Mono3D DRAM Tiers

**Costs.** The scheduler ([第 3.1 節](#section-3-1)) may trigger expert swaps *between batches*. To evaluate the worst-case scenario, we consider 1) short sequences, ${L_{\mathrm{in}}}={L_{\mathrm{out}}}=256$ with batch size one, and 2) consecutive batches assigned to different topics. [Table 4](#table-04) reports the time and energy overheads of expert swaps, which remain well below 1% across all benchmarks. This negligible cost stems from two factors: expert swaps occur within the same bank, avoiding cross-bank movement, and NMP logic includes dedicated row-swap buffers that enables swapping at the high internal Mono3D DRAM tier bandwidth without traversing the DRAM–xPU interface.

<span id="section-6-3-4"></span>

#### 6.3.4 バッチサイズによる性能スケーリング

<span id="figure-18"></span>

![図 18. Impacts of (a) batch size and (b) Mono3D DRAM layers on system-level metrics, evaluated with Llama-4-Scout on Stratum-XL](../../papers/stratum/figure-18.png)

**図 18.** Impacts of (a) batch size and (b) Mono3D DRAM layers on system-level metrics, evaluated with Llama-4-Scout on Stratum-XL

[図 18](#figure-18)(a) evaluates Stratum’s performance scaling across different query batch sizes using the large-scale Llama-4-Scout [Lla25] benchmark. Batch sizes are chosen to ensure the full model fits within the Mono3D DRAM of Stratum or the HBM of the GPU baseline. Stratum consistently outperforms the GPU baseline across all settings by 4.7–$9.8\times$. However, the relative performance advantage reduces with larger batches, particularly at shorter sequence lengths (e.g., 1024 tokens), due to the GPU die’s higher compute-to-bandwidth ratio and the increased dominance of MoE layers in the overall runtime.

<span id="section-6-3-5"></span>

#### 6.3.5 Mono3D DRAM 層数による性能スケーリング

[図 18](#figure-18)(b) reports Stratum’s performance scaling across different Mono3D DRAM layer configurations. All variants have the same DRAM capacity and use the same NMP logic die processor, and throughput is normalized to the die area of each Mono3D DRAM to ensure a fair, cost-aware comparison. On average, the 1024-layer design achieves $1.21\times$ and $2.96\times$ higher throughput per area than the 256-layer and 64-layer Mono3D DRAM, respectively, demonstrating the cost-efficiency benefits of adopting >1k-layer Mono3D DRAM.

<span id="section-6-3-6"></span>

#### 6.3.6 少ない層数の Mono3D DRAM における階層化機構

提案する階層化機構は、モノリシック 3D DRAM の垂直積層から生じるワード線遅延の変動を利用する。Mono3D DRAM は 400 層を超えてスケールした 3D NAND Flash と同様の製造工程を採用する [A25]。そこで NMP ロジック設計を維持したまま、元の 1024 層 mat を水平接続された 512 層セグメント 2 個に分割し、512 層構成を検討する。デバイスレベルシミュレーションでは最速層と最遅層のアクセス遅延に $1.3\times$ の差が現れる。${L_{\mathrm{in}}}={L_{\mathrm{out}}}=1024$ の系列長で LLama-4-Scout [Lla25]、Mixtral $8 \times 7$B [Jia24a]、OLMoE-1B-7B [Olm24] を評価すると、トピック対応階層化配置により MoE と注意層を含む全体性能がそれぞれ 17.7%、18.3%、18.3% 向上する。これらの結果は、多様な Mono3D DRAM 層数で提案階層化戦略が有効であることを示す。

<span id="section-7"></span>

## 7 関連研究

**3D Stackable DRAM.** Monolithic 3D-Stackable DRAM has emerged as a promising alternative to HBM by sequentially fabricating multiple DRAM layers on the same wafer. Unlike HBM, which depends on TSVs and costly die-stacking, Mono3D DRAM employs fine-pitch hybrid bonding for higher internal bandwidth and integration density [Ong23, A23, A22b, A23a, Sig25, Mon25]. Leading Mono3D DRAM technologies include Horizontal 1T1C [Ong23, A23], which reorients and stacks 1T1C DRAM cells, and Gate-Control Thyristors [A22b, A23a], which leverage avalanche mechanisms. Recent work further shows that Mono3D DRAM ’s $\sim$1$\mu$m bonding pitch [Che20a] enables up to $5\times$ denser vertical interconnects than HBM [Exp24].

**Processing In/Near Memory Acceleration for Transformers.** While Processing In/Near Memory (PIM/PNM) has been a long-standing concept, MAT [Mat21a] first applied PIM to Transformer models, targeting a single encoder block with a memory-efficient pipelined sub-sequence flow. TransPIM [Tra22] extends this with a hybrid PIM-PNM architecture for full-model execution. Neupims [Neu24a] and AttAcc [Att24] focus on Decoder-only Transformer models, offloading attention layers in the decoding stage to the PNM on a xPU-PNM hybrid-processing system. Duplex [Dup24] further expanded support to MoE, GQA, and continuous batching with dynamic compute partitioning. However, all these designs rely on 2D DRAM or die-stacked HBM, limiting their effectiveness when applied to Mono3D DRAM-based systems.

<span id="section-8"></span>

## 8 結論

Stratum は効率的な MoE サービスのための新しいシステム・ハードウェア協調設計であり、3D ハイブリッドボンディングで高密度 Mono3D DRAM ダイをロジックへ統合し、さらに 2.5D シリコンインターポーザで GPU へ接続する。この構成は従来の GPU–HBM システムに対し、低コストで高スループットな代替を提供する。ハードウェアレベルでは、Mono3D DRAM の垂直アクセス遅延差を利用するメモリ階層化と、エキスパート・注意計算に最適化した近メモリプロセッサ (NMP) を導入する。システムレベルでは、トピック依存のエキスパート活性パターンを利用してエキスパートを分類・メモリ層へマッピングし、軽量分類器に導かれるトピック対応スケジューラでサービス目標を満たす。デバイス、回路、アルゴリズム、システムにまたがる評価により、GPU ベースラインと比べて解読スループットが最大 $8.29\times$、エネルギー効率が最大 $7.66\times$ 改善することを示した。

謝辞。本研究の一部は、DARPA が支援する SRC プログラム JUMP 2.0 のセンターである PRISM と CoCoSys の支援を受けた。また、米国国立科学財団 (NSF) の助成金 2112665、2112167、2003279、2120019、2211386 の支援を受けた。

[+equal contribution]: Yue Pan and Zihan Xia contributed equally.
