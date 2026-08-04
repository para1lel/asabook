---
title: 'TileSight: Tile-Centric GPU Performance Model'
createTime: 2026/08/04 08:48:07
permalink: /ja/papers/tilesight/
pageClass: tilesight-paper
---

> [Zhiwen Mo](https://hamerlate.github.io/)、[Yu Cheng](https://chengyupku.github.io/)、[Lei Wang](https://dblp.org/pid/181/2817-222)、[Zhengju Tang](https://dblp.org/pid/371/5817)、[Lei Xu](https://orcid.org/0000-0002-6226-3063)、[Guoyu Li](https://dblp.org/pid/61/8379)、[Yuqi Dong](https://dblp.org/pid/294/5118)、[Lingxiao Ma](https://xysmlx.github.io/)、[Yuqing Xia](https://dblp.org/pid/211/8365)、[Jilong Xue](https://dblp.org/pid/06/10336)、[Fan Yang](https://fanyangcs.github.io/)、[Luo Mai](https://luomai.github.io/)、[Zhi Yang](https://yangzhihome.github.io/)、[Wayne Luk](https://profiles.imperial.ac.uk/w.luk)、[Hongxiang Fan](https://os-hxfan.github.io/)。2026 年 7 月 24 日に arXiv へ初投稿。本稿は [arXiv ページの version 1、*TileSight: A First-Principles Tile-Centric Analytical GPU Performance Model from Cores to Clusters*](https://arxiv.org/abs/2607.22432) を転記・翻訳したリーディング版です。[原論文 PDF](/paper/tilesight.pdf)、[arXiv DOI](https://doi.org/10.48550/arXiv.2607.22432)、[TeX ソース](https://arxiv.org/src/2607.22432) も参照できます。厳密な印刷レイアウトと参考文献については原論文 PDF を正とします。

## 概要

Triton、TileLang、CUDA Tile などの近年の GPU プログラミング フレームワークは、tile を第一級の言語プリミティブとして採用しており、tile 中心プログラミングは高性能 GPU kernel を記述する主流の方法となっています。しかし、tile ベース プログラムの性能分析ツールはこの流れに追随していません。プログラマは、kernel が実際にどのように動作するかを推論するために、依然として粗い Roofline 境界、不透明な ML ベース予測器、事後的な profiler に頼っています。この隔たりは、kernel fusion と分散推論が Tensor Core、CUDA core、cache 階層、memory pipeline、GPU 間 network の相互作用に左右される現代の AI workload にとって、ますます深刻になっています。私たちは、tile をプログラミング プリミティブから分析プリミティブへと発展させた tile 中心性能モデリング ツール TileSight により、この隔たりを埋めます。単一 GPU core 内では、TileSight は compute pipeline と memory pipeline の重なりをモデル化します。同一 chip 上の core 間では、TileSight は cache 階層をモデル化します。GPU 間では、node 間通信をモデル化します。3 層はすべて tile 抽象を共有します。*1)* tile 内層は、各 tile の仕事を network、memory、compute pipeline にまたがる resource vector として表します。*2)* tile 間層は、依存関係と順序を持つ tile action をスケジュールして可能な重なりを明らかにし、tile reuse distance から多段 cache の hit rate を推定します。*3)* device 間層は、remote tensor access を placement に対応付け、$\alpha$-$\beta$ stage cost によって routing します。A100、H200、B200、B6000 で評価したところ、TileSight は測定された単一 GPU kernel latency を pooled mean absolute percentage error (MAPE) 12.35% 以内で再現し、最先端 baseline を上回るとともに、4 つの architecture 間でより優れた転移性を示しました。各 GPU における L2 cache hit rate の予測値は、測定値からおよそ 1 percentage point 以内でした。32 GPU deployment まで拡張すると、TileSight は fused distributed kernel で weighted MAPE (wMAPE) 16.18%、end-to-end vLLM serving で 13.52% wMAPE を達成しました。最適化 loop に組み込んだ場合、TileSight は本稿の case study において強力な vendor baseline および expert baseline と競合する tile configuration を選択します。TileSight は論文公開時に open source 化される予定です。

## 1 はじめに

Large language model (LLM) の規模拡大により、training system と serving system は hardware limit にますます近づき、kernel efficiency が latency と cost の両方を左右する中心的要因となっています。この性能を引き出すため、開発者は複数の LLM operation を大規模な tile program へと融合するようになっており、その bottleneck は tiling、memory movement、pipeline overlap、wavefront scheduling によって決まります。したがって、performance boundary を明らかにして最適化を導くには、正確かつ高速な white-box performance modeling が必要です。

Kernel 最適化を容易にするため、GPU programming community は共通の paradigm として *tile 中心プログラミング*へ収束してきました。Triton [Til19] は tile-level load、store、dot product を切り開き、PyTorch の custom kernel における事実上の標準となりました。TileLang [Til25g] はさらに、tile level で dataflow と scheduling を分離します。NVIDIA の CUDA Tile [Nvi26] (CUDA 13.1、2025) は、およそ 20 年で最も重要な CUDA の進歩 [Fut26] と説明され、tile を programming primitive として正式に採用しています。一方、CuteDSL [NviCut] は CUTLASS の tile abstraction を Python domain-specific language (DSL) として公開します。この結果、tile は現代の GPU programming における中心的 abstraction となりました。しかし、**性能分析はこの tile 中心 abstraction に追随していません**。Triton は数千 configuration の black-box autotuning に依存し [Til19]、Roofline model [Wil09] は L2 cache miss と shared memory bank conflict を区別できず、ML-based predictor [Lee25, Geo21] は architecture ごとの training を必要とし不透明です。Nsight Compute (NCU) などの profiler や profiling-based tool [Gua25, Hua25] は事後的です。instrumentation や clock の変更によって実行を乱す可能性があり、その counter report は、観測された bottleneck をどの tile、reuse pattern、pipeline stage が引き起こしたかを説明しません。[表 1](#table-01) はこの abstraction mismatch をまとめています。Tile 中心プログラミングの採用が進むにつれ、kernel を実行せずに tile-configuration の変更が性能へ及ぼす影響を予測できる、正確で効率的な *tile 中心 performance model* が強く求められています。

このような model の必要性は、tiled kernel の main loop の*内部*で起こることを考えると、さらに明確になります。GEMM でさえ、性能は FLOP 数と byte 数だけでなく、software-pipeline depth、streaming multiprocessor (SM) あたりの resident tile 数、load-compute overlap に依存します。Fused kernel では問題がさらに顕著になります。[図 1](#figure-01) に示す H100 上の FlashAttention-3 (FA-3) は、Tensor Core 上の 2 つの GEMM、CUDA core 上の reduction と softmax、special function unit (SFU) 上の special function を含む 10 種類以上の異なる operation と、その間の細粒度な data dependency を持ちます。これらの operation は異なる hardware resource を占有しており、重なる可能性がありますが、その度合いは scheduling order と pipeline depth に決定的に依存します。Roofline、profiler、autotuner を含む既存 tool は、この tile 内 scheduling structure をほとんど認識できません。さらに、これらの複雑な kernel は*分散*環境でますます必要とされています。たとえば、tensor parallelism (TP)、expert parallelism (EP)、sequence parallelism (SP) は workload を複数 GPU に partition し [Sve25]、compute と重ねる必要がある collective communication を導入します。Distributed kernel の性能は、global tile grid の partition 方法、使用する communication primitive、compute pipeline と communication pipeline の interleave 方法に依存します。これらの判断は現在、直感または高価な trial-and-error によって行われています。

これらの課題に対し、tile は GPU system の performance modeling に*自然な第一級 abstraction*をもたらすと私たちは考えます。その理由は 3 つの性質にあります。**(1) 決定性**: tile configuration (shape、pipeline depth、memory layout) が与えられると、各 tile の resource usage は完全に決まり、simulation なしで analytical modeling が可能です。**(2) 合成可能性**: tile information は階層的に合成できます。各 tile は pipeline ごとの resource decomposition (tile 内) を持ち、tile は dependency、concurrent issue、execution order (tile 間) により関連し、tile grid は placement (device 間) を通じて device をまたいで拡張されます。各 level を個別に model 化してから合成できます。**(3) 可搬性**: tile abstraction は多様な GPU architecture で採用されています (本稿では NVIDIA A100、H100、H200、B200、RTX PRO 6000 Blackwell (B6000)、AMD MI210 を扱います)。すべての現代 GPU は類似した階層 memory と compute structure によって tile-shaped workload を実行するためです。

これらの知見に基づき、私たちは *統一 tile 中心 analytical execution engine* である TileSight を提案します。性能を単一の bottleneck resource に帰属させる Roofline model とは異なり、TileSight は tile execution plan が hardware 上で展開される様子を analytical に simulate し、実際の kernel 性能を決める prologue、steady-state overlap、epilogue structure を捉えます。この simulation は、統一された tile-based abstraction によって 3 つの階層 level を合成します。

- **Tile 内**: 各 tile は operation、src/dst placement descriptor、footprint によって特徴付けられ、それらから network、memory、compute にまたがる独立 scheduling 可能な hardware pipeline 上の時間へ仕事を分解する tile ごとの *resource vector* が得られます。同じ placement descriptor が、fusion (intermediate を register または shared memory (SMEM) に保持) と device 間 movement を統一します。
- **Tile 間**: tile は producer-consumer dependency、concurrent issue、execution order によって関連付けられます。これらは、tile-action directed acyclic graph (DAG) 上で topological-order search を駆動し、fused kernel body 内で最良の合法的な pipeline overlap を選びます。また、stochastic distance-based cache modeling (SDCM) を用いた多段 *tile reuse distance* analysis により、grid traversal から暗黙の cache hit rate を導出します。
- **Device 間**: device 間実行は同じ tile 内 abstraction の placement case です。source または destination が device をまたぐ tile は、基礎となる remote tensor access の routed $\alpha$-$\beta$ cost から計算された `Net` entry を得るだけで、同じ envelope を適用できます。

重要なのは、この 3 level が共有 core abstraction によって共同設計されていることです。`HardwareUsage` は pipeline ごとの time decomposition、*tile action* は合成可能な scheduling unit、`TileGrid` は workload descriptor です。まとめると、本稿の貢献は次のとおりです。

**(1) 統一 tile 中心 analytical execution engine**。Tile execution plan が hardware pipeline 上でどのように展開されるかを simulate し、tile ごとの resource decomposition (tile 内)、dependency-driven DAG ordering と tile reuse-distance cache modeling (tile 間)、placement-based device 間 tile access を、共有 abstraction を持つ 1 つの framework で扱います (第 3 節)。

**(2) Tile-pipeline overlap analysis**。通常の software-pipelined loop (GEMM の load-compute overlap など) と複雑な fused kernel (FlashAttention や multi-head latent attention (MLA) decode など) の両方を、dependency-constrained tile-action DAG 上の反復 tile pipeline として model 化します。Pipeline depth、resident tile interleaving、合法的 tile-action ordering を組み合わせることで、TileSight は単純な Roofline model が見落とす prologue、steady-state、epilogue cost を予測します (第 3.4 節)。

**(3) Tile reuse-distance cache modeling**。Cache behavior を独立した trace-simulation 問題ではなく、tile execution plan の自然な帰結とします。GPU schedule と同じ granularity で reuse を推論することで、TileSight は analytical performance model 内で高速かつ schedule-sensitive な multi-level cache modeling を実現し、軽量な approximation と sampling technique により accuracy を維持します (第 3.5 節)。

**(4) Tile placement による合成可能な distributed extension**。Device 間実行を同じ tile abstraction の placement case として扱います。Remote tensor access を producer-consumer placement から推定し、logical exchange の順序付き stage へ分解します。その routed $\alpha$-$\beta$ cost が tile ごとの resource vector の network entry を埋めるため、device 間 movement は同じ envelope を介して local compute と合成されます (第 3.6 節)。

## 2 背景と動機

### 2.1 GPU 性能モデリング

<span id="figure-01"></span>

![H100 上での FlashAttention-3 の実行](../../papers/tilesight/figure-01.png)

**図 1.** H100 上の FlashAttention-3。(a) Tensor Core、CUDA core、SFU にまたがる 10 種類以上の heterogeneous operation。(b) その data-dependency DAG。(c) scheduling order が compute-memory pipeline overlap を決める仕組み。

<span id="figure-02"></span>

![L2 bandwidth と working-set size の関係](../../papers/tilesight/figure-02.png)

**図 2.** B200 と B6000 における L2 bandwidth と working-set size の関係。multi-level cache hierarchy が明らかになります。B200 (dual-die) は約 ${\sim}22.5$ TB/s の level-1.5 (L1.5)/LRC tier と約 ${\sim}83$ MB の緩やかな L2 cliff を示し、B6000 (single-die) は約 ${\sim}130$ MB に鋭い cliff を示します。TileSight はこれらの sweep により GPU ごとの effective cache capacity を calibration します。

既存の GPU performance tool は 3 つの category に分類できます。**Learned predictor と hybrid predictor** [Lee25, Zha26p] は architecture ごとの trace から end-to-end runtime または analytical-model residual を fit し、純粋な analytical model では現代 GPU の相互作用の複雑さを捉えられないと主張します。しかし、いずれも retraining が必要で interpretability は限定的です。**Analytical model** [Wil09, Par19, Zhe23] は portable で explainable ですが、通常は GPU execution を aggregate compute term と bandwidth term に集約します。**Profiling tool と simulation tool** [Gua25, Hua25, Agr24, Wan25s] は実行後の測定 behavior を示しますが、kernel を再実行する前に tile shape、pipeline depth、swizzle の変更がどのように動作するかを予測しません。第 1 節で述べたように、共通の限界は abstraction mismatch です。これらの tool は、現代の GPU program が使用する tile granularity で性能を model 化しません。特に hybrid predictor がこの複雑さを learned component に委ねるのに対し、TileSight は first-principles tile-centric simulation によって、完全な white-box のままそれを捉えられることを示します。

### 2.2 Tile 中心プログラムにおけるモデリングの隔たり

不足している abstraction は 3 level に現れます。**Tile 内**: 各 tile は compute、memory、network にまたがる heterogeneous pipeline を使用するため、単一 bottleneck scalar では overlap を決める per-pipeline structure を捉えられません ([図 1](#figure-01))。**Tile 間**: tile dependency は fused body 内の合法的 action ordering を決め、grid 上の tile execution order は cache reuse を決めます。現代 GPU では単一の flat bandwidth 値では不十分です ([図 2](#figure-02))。**Device 間**: partition された tile grid は communication pipeline を通じて data を交換し、その時間は独立に加算するのではなく compute と overlap させる必要があります。[表 1](#table-01) は、既存 tool がこれらの level の 1 つ以上を欠く様子をまとめています。

<span id="table-01"></span>

<div class="paper-wide-table">

| 機能 | Roofline [Wil09] | NeuSight [Lee25] | PipeWeave [Zha26p] | GenZ [Bam24] | Vidur [Agr24] | SimAI [Wan25s] | TileSight |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Kernel profiling/training 不要¹ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✓ |
| Pipeline-aware² | ✗ | ✗ | ○ | ✗ | ✗ | ✗ | ✓ |
| Cache-aware³ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| 明示的な fused program⁴ | ✗ | ✗ | ○ | ✗ | ✗ | ✗ | ✓ |
| Distributed⁵ | ✗ | ○ | ○ | ○ | ✓ | ✓ | ✓ |
| Compute-communication overlap⁶ | ✗ | ✗ | ✗ | ✗ | ✗ | ○ | ✓ |
| Interpretable⁷ | ✓ | ✗ | ○ | ✓ | ✗ | ✗ | ✓ |

</div>

**表 1.** 既存 performance modeling tool との比較。✓ 完全対応。○ 部分対応。✗ 非対応。¹Kernel profiling/training 不要: kernel execution trace や ML training は不要です。TileSight が使用するのは architecture ごとに一度だけ行う microbenchmark (bandwidth/throughput/latency sweep、約数分) のみです。²Pipeline-aware: tile 内 DAG scheduling と compute-memory pipeline overlap。³Cache-aware: L2/L1.5 hit rate と schedule-dependent tile locality effect を予測します。⁴明示的な fused program: 固定された対応 pattern に限定されず、ユーザーが記述する任意の multi-op DAG kernel (FA-3、MLA など) を扱います。⁵Distributed: multi-GPU collective communication modeling。⁶Compute-communication overlap: fused compute-communication kernel (AllGather+GEMM など)。SimAI はユーザー指定 overlap ratio を受け取りますが、analytical には導出しません。⁷Interpretable: bottleneck diagnosis を支援する white-box model。

## 3 階層的 Tile-Pipeline モデル

TileSight は *tile* を第一級の modeling unit として扱い、prologue-steady-epilogue pipeline envelope を program の各 level へ再帰的に適用します。Tile は *tile 内* information (operation、src/dst placement、footprint、および独立 scheduling 可能な各 hardware pipeline で占有する resource) を持ち、*tile 間* relationship (producer-consumer dependency、concurrent issue、loop、tile grid、wave にまたがる execution order) に参加します。したがって、tiled workload は *tile execution plan*、すなわちこの 2 種類の information で注釈された tile graph です。Distributed execution も同じ tile-based abstraction を共有します。source または destination が device をまたぐ tile は resource vector に `Net` entry を加えるだけで、同じ envelope を適用できます。

### 3.1 Workload から Tile Execution Plan へ

TileSight への入力は、tiled GEMM、fused attention kernel、all-gather に続く GEMM、GPU 間の Mixture-of-Experts (MoE) routing などの high-level workload です。Workload は tensor とその placement を固定しますが、schedule は未指定のままです。TileSight はそれを tile execution plan へ引き上げ、performance modeling に必要な schedule-relevant choice、すなわち tile shape、loop と reduction の順序、block swizzle、software-pipeline depth、SM あたりの resident block 数、distributed partitioning、collective implementation を明示します。Triton や TileLang などの tile-centric DSL は、この information の大部分を直接公開します。手書き kernel では、同じ field を kernel schedule から手動で与えます。

<span id="figure-03"></span>

![TileSight の設計概要](../../papers/tilesight/figure-03.png)

**図 3.** All-gather-GEMM (AG-GEMM) における **TileSight の設計概要**。**(a)** workload は operator と tensor placement のみで記述されます ($X$ は $N$ GPU に column-shard)。**(b)** TileSight はそれを、DAG が memory level $L_0$-$L_4$ にまたがる tile schedule へ引き上げます。**(c)** 単一の hardware abstraction が register、SMEM、L2、HBM、GPU 間 fabric を 5-level hierarchy として公開します。**(d)** tile 内 resource vector と tile 間 DAG/concurrency analysis が、再帰的 prologue-steady-epilogue envelope への入力となります。**(e)** engine は envelope を timeline として描画します。software-pipelined load は compute と overlap し、AllGather は `Net` lane 上で *placement から推定*されます。**(f)** latency、utilization、cache hit、overlap rate を含む tile ごとの performance report。

<span id="table-02"></span>

| Field | 側 | Model における役割 |
| --- | --- | --- |
| Tensor access | Tile 内 | tile ごとの footprint と、tensor tile がどこで生成され、どこに存在するかを記録する placement descriptor。register、architecture-specific tensor memory (TMEM)、SMEM、local cache/DDR、または GPU group 上の shard/replica。Reuse dimension は tile 間 cache modeling への入力となります。 |
| Operation type | Tile 内 | Tile が行う action。load、store、tensor-core または CUDA-core matmul、reduction、exponential、rescaling、remote transfer、fused composite。 |
| Resource vector | Tile 内 | 独立 scheduling 可能な resource (tensor core (TC)、CUDA core、SFU、TMEM、SMEM、L1.5、L2、DDR、Net) 上での tile ごとの時間。operation、footprint、placement、calibration 済み hardware rate から導出されます。 |
| Tile grid | Tile 間 | spatial tile shape、launch order、swizzle、loop/reduction depth、distributed partition。tile execution order、wave、device ごとの local work を決めます。 |
| Producer-consumer DAG | Tile 間 | tensor の production と consumption に基づく tile 間 edge。placement とともに、iteration 内の合法的 ordering を決めます。 |
| Concurrency と depth | Tile 間 | software-pipeline stage、SM あたりの resident block、同時 issue 可能な tile。effective pipeline depth を設定します。 |

**表 2.** Tile execution plan は、各 field が単独の tile (tile 内) を記述するか、tile 間の relationship (tile 間) を記述するかによって field を分類します。

この plan は意図的に thread-level detail を避けます。Tile programmer と distributed runtime が実際に変更する choice だけを保持し、それらの choice が、どの tile が pipeline に入り、どの resource を占有し、互いにどう依存または並行実行するかを変えます。その後、cache traffic、wave effect、communication stage、pipeline overlap は個別に加えるのではなく、これらから導出されます。

### 3.2 Tile 内: Tile とその Resource Vector

Tile は、*operation* (load、store、tensor-core または CUDA-core matmul、reduction、exponential、rescaling、remote transfer、fused composite)、*footprint* (tile ごとの byte と FLOP)、および input が生成される場所と output が存在する場所を記録する *src*/*dst placement descriptor* によって特徴付けられます。場所は register、architecture-specific tensor memory (TMEM)、shared-memory scratchpad、L1.5 または L2 cache、local device の DDR、GPU group 上の shard/replica のいずれかです。Placement は、同じ tile 内 representation で fusion と device 間 movement の両方を表すための中心的 abstraction です。Intermediate output を register、TMEM、SMEM scope として mark すると global-memory store が除去され (fusion)、load source を remote shard として mark すると load が device 間 transfer になります (distribution)。

TileSight は各 tile について、これらの property を独立 scheduling 可能な hardware resource 上の時間 vector へ変換します。

$$
\mathbf{u}(o)=
\langle t_{\mathrm{TC}}, t_{\mathrm{CUDA}}, t_{\mathrm{SFU}}, t_{\mathrm{TMEM}},
t_{\mathrm{SMEM}}, t_{\mathrm{L1.5}}, t_{\mathrm{L2}}, t_{\mathrm{DDR}}, t_{\mathrm{Net}}\rangle .
\tag{1}
$$

これは tile の operation、footprint、src/dst placement、および one-shot microbenchmark で calibration した rate から計算されます。Pure tensor-core matmul tile は TC entry のみを埋めます。Blackwell attention tile は softmax と correction の load/store について明示的な TMEM traffic も加算します。DDR からの load tile は DDR を埋めます (access が cache に hit する場合は L1.5/L2 も埋めます)。remote-load tile は Net を埋めます。この vector は Roofline scalar より表現力があります。異なる pipeline 上の tile は overlap できる一方、同じ pipeline を競合する tile は serialize され、remote movement は同じ仕組みによって local compute と合成されるためです。$\mathbf{u}(o)$ のうち 2 つの entry は、単独の tile では固定されません。Memory tile の L1.5/L2/DDR split は access が cache に hit するかどうかに依存し、第 3.5 節の tile reuse distance から導出されます。Remote tile の `Net` entry は、基礎となる communication stage の routed cost に依存し、第 3.6 節で導出されます。Algorithm 1 は、すべての component が master loop に接続される方法を概略します。以降の subsection で各 block を詳述します。

**Algorithm 1: 階層的 Tile-Pipeline 評価。**

- **入力:** tile execution plan $P$、hardware specification $H$、optional distributed mapping $\Pi$。
- **出力:** predicted latency $T$ と resource ごとの utilization。
- $G \leftarrow P$ の tile grid、launch order、swizzle。
- $A \leftarrow P$ の tensor access、reuse dimension、placement descriptor。
- $D \leftarrow P$ の tile-action DAG。
- $S \leftarrow P$ の software-pipeline parameter。
- **もし** $\Pi$ が空でなければ:
  - $G,A,D \leftarrow \mathrm{PartitionTilePlan}(G,A,D,\Pi)$。single device は local-only case。
  - $\mathcal{O}_{\mathrm{net}} \leftarrow \mathrm{InferRemoteTensorAccesses}(G,A,D,\Pi)$。
  - $\mathcal{N} \leftarrow H$ の network topology と calibration 済み $\alpha,\beta$ parameter。
  - **各** remote tensor access sequence $c \in \mathcal{O}_{\mathrm{net}}$ **について**:
    - $\mathcal{K}_c \leftarrow \mathrm{DecomposeIntoStages}(c)$。ring step や tree level など。
    - **各** stage $k \in \mathcal{K}_c$ **について**:
      - $\mathcal{E}_k \leftarrow \mathrm{LogicalExchanges}(k)$。tuple $(\mathrm{src},\mathrm{dst},\mathrm{bytes})$。
      - $\mathcal{R}_k \leftarrow \mathrm{Route}(\mathcal{E}_k,\mathcal{N})$。
      - $T_k,U_k \leftarrow \mathrm{AlphaBetaStageTime}(\mathcal{R}_k,\mathcal{N})$。
    - $D$ 内の対応する transfer tile を、`Net` 上の $\sum_k T_k$ で annotate。
- $C \leftarrow \mathrm{CacheTraffic}(G,A,H)$。
- $D$ 内の memory tile を $C$ の L1.5/L2/DDR entry で annotate。
- $p \leftarrow \mathrm{ResidentTilesPerSM}(P,H)$。
- $d \leftarrow S.\mathrm{stages}\times p-1$。
- $\mathcal{E}_{\mathrm{tile}} \leftarrow \mathrm{PipelineEnvelope}(D,d,H,\mathrm{active\ SMs})$。
- $T,U \leftarrow \mathrm{WaveAggregate}(G,\mathcal{E}_{\mathrm{tile}},H)$。
- **返す** $T,U$。

### 3.3 Tile 間: Dependency、Concurrency、Order

Tile は 3 種類の tile 間 information によって接続されます。*1)* *Producer-consumer dependency* は iteration 内の合法的 ordering を固定します。FlashAttention では、$Q$/$K$ load は gemm1 ($Q\!@\!K$) より前、gemm1 は softmax より前、softmax は gemm2 ($P\!@\!V$) より前です。Placement および dependency とともに、どの intermediate が register/TMEM/SMEM に残り、どれが global memory に spill するかを決めます。*2)* *Concurrent issue* は、resource vector が競合しない dependency のない tile を同時に実行できるようにします。たとえば、attention の次の $K$-block を load しながら現在の block を compute したり、同じ $K$ slice に沿って GEMM の A load と B load を同時に issue したりできます。同じ tile 集合でも複数の合法的 order があり、shared pipeline 上で異なる overlap を生じます。*3)* loop iteration と tile grid にまたがる *tile execution order* は、どの load が cache に resident data を見つけるかを決めます。row-panel traversal は隣接 $M$-row の B-tile reuse を維持し、block swizzle は sequence を並べ替え、persistent-block schedule は tile を SM に固定します。この 3 要素が、pipeline envelope に必要な入力そのものです。

### 3.4 Pipeline Envelope: Prologue-Steady-Epilogue

Resource vector と tile 間 relationship を持つ tile 集合が与えられると、TileSight は execution を pipeline として評価します。$N$ 回の logical iteration と effective depth $d$ を持つ repeated unit について:

$$
T =
T_{\mathrm{pro}} +
\max(N-d,0)\,T_{\mathrm{steady}} +
T_{\mathrm{epi}},
\tag{2}
$$

ここで $T_{\mathrm{pro}}$ は fill cost、$T_{\mathrm{steady}}$ は repeated unit ごとの overlapped cost、$T_{\mathrm{epi}}$ は drain cost です。同じ envelope が tile execution plan の各 level に再帰的に適用されます。outer envelope (tile-block wave 全体) の steady-state body 自体が pipeline ($K$-loop 全体) となり、その steady body がさらに inner action sequence 全体の pipeline となり得ます。Effective depth は、明示的 software-pipeline stage と resident tile interleaving を組み合わせます。

$$
d = \mathrm{stages} \times \mathrm{resident\_tiles\_per\_SM} - 1 .
\tag{3}
$$

したがって、SM あたり 2 block の schedule は特別な case ではありません。1 つの resident tile-block が memory を待っている間に、SM が別の tile-block から work を issue できるため、pipeline を深くします。

**Steady-state overlap。** Tile sequence の steady-state cost は、どの合法的 ordering を選ぶかに依存します。式 1 の同一 hardware dimension を使用する tile はその dimension 上で累積し、独立 dimension は overlap するためです。

$$
T_{\mathrm{steady}}(\sigma)
=
\max_{r}
\sum_{o \in \sigma} u_r(o),
\tag{4}
$$

これは DAG 内のすべての data-dependency edge に従います。選択される steady state は最良の合法的 ordering です。

$$
T_{\mathrm{steady}} =
\min_{\sigma \in \mathrm{Topo}(D)} T_{\mathrm{steady}}(\sigma).
\tag{5}
$$

実際には、real fused-kernel DAG は強く制約されるため、この search は小規模です。MLA decode の 11 tile action は、制約なしの $11!$ permutation から 132 の合法的 topological order へ減少します。この search は autotuning run ではありません。Tile plan 上の analytical scheduling step であるため、cost model 内で実行できるほど低 cost のままです。

**Boundary cost。** Prologue と epilogue は同じ resource vector から計算されますが、overlap は少なくなります。load-compute pipeline では、prologue は主に pipeline を fill する memory tile で構成され、epilogue は残りの compute と final store で構成されます。Fused tile body は、一方または両方の boundary に reduction または normalization を加えます。この分離は重要です。loop count が短い場合や launch される wave が少ない場合、同じ steady-state bottleneck を持つ 2 つの schedule でも end-to-end time が異なる可能性があるためです。

**Resident tile と wave。** Occupancy は utilization だけでなく overlap structure も変えます。1 SM に $p$ 個の tile-block が resident であれば、model はそれらを同じ tile pipeline の interleaved instance として扱います。resident count は shared memory、register、warp limit、architecture-specific maximum blocks per SM によって制限されます。同じ wave decomposition が tail effect も扱います。tail wave は SM の一部だけを使用することがあり、active SM は shared L2/DDR bandwidth のより大きな share を得るため、その active-SM count を用いて envelope を再計算します。Algorithm 2 はこの評価を展開し、tile loop structure を再帰的にたどって dependency-valid ordering を列挙します。

**Algorithm 2: 再帰的 Pipeline-Envelope 評価。**

- **関数** $\mathrm{OverlapAnalysis}(P,H)$:
  - $p \gets \mathrm{ResidentTilesPerSM}(P,H)$。
  - $(n_{\mathrm{full}}, n_{\mathrm{tail}}) \gets \mathrm{WaveDecompose}(P.\mathrm{grid}, H.\mathrm{SMs}, p)$。
  - $(T^{\mathrm{full}}, U^{\mathrm{full}}) \gets \mathrm{AnalyzeLoop}(P.\mathrm{root}, p, H.\mathrm{SMs})$。
  - **もし** $n_{\mathrm{tail}} > 0$:
    - $(T^{\mathrm{tail}}, U^{\mathrm{tail}}) \gets \mathrm{AnalyzeLoop}(P.\mathrm{root}, p, n_{\mathrm{tail}})$。
  - **それ以外:**
    - $T^{\mathrm{tail}} \gets 0,\ U^{\mathrm{tail}} \gets \emptyset$。
  - **返す** $n_{\mathrm{full}}T^{\mathrm{full}} + T^{\mathrm{tail}},\ \mathrm{MergeMetrics}(U^{\mathrm{full}},U^{\mathrm{tail}})$。
- **関数** $\mathrm{AnalyzeLoop}(\mathrm{node}, \mathrm{stage}, \mathrm{active\_SMs})$:
  - $\mathrm{groups} \gets \mathrm{GetSubNodes}(\mathrm{node})$。
  - **もし** $\mathrm{node}$ が inner loop なら:
    - $s \gets \mathrm{GetPipelineStage}(\mathrm{node})$。
    - $d \gets s \times \mathrm{stage} - 1$。software stage $\times$ resident tile。
    - **返す** $\mathrm{ModelOverlap}(\mathrm{groups},d,\mathrm{active\_SMs})$。
  - $\mathrm{metrics} \gets [\,]$。
  - **各** $g \in \mathrm{groups}$ **について**:
    - **もし** $g$ が loop なら:
      - $\mathrm{metrics.append}($ $\mathrm{AnalyzeLoop}(g,\mathrm{stage},\mathrm{active\_SMs}))$。
    - **それ以外:**
      - $\mathrm{metrics.append}($ $\mathrm{ModelOverlap}([g],\mathrm{stage}-1,\mathrm{active\_SMs}))$。
  - **返す** $\mathrm{MergeMetrics}(\mathrm{metrics})$。
- **関数** $\mathrm{ModelOverlap}(\mathrm{groups},d,\mathrm{active\_SMs})$:
  - $N \gets \mathrm{groups}$ が表す repeated-iteration count。
  - $\mathrm{best} \gets \infty$。
  - **各** $\sigma \in \mathrm{Topo}(\mathrm{groups})$ **について**:
    - $\mathbf{u}_{\sigma} \gets$ order $\sigma$ と $\mathrm{active\_SMs}$ の下での resource-vector accumulation。
    - $T_{\mathrm{pro}},T_{\mathrm{steady}},T_{\mathrm{epi}} \gets \mathbf{u}_{\sigma}$ から得た boundary cost と steady cost。
    - $T \gets T_{\mathrm{pro}}+\max(N-d,0)T_{\mathrm{steady}}+T_{\mathrm{epi}}$。
    - **もし** $T < \mathrm{best}$: $\mathrm{best} \gets T$。
  - **返す** $\mathrm{best}$ と対応する utilization。

### 3.5 Tile Reuse Distance による Cache Traffic

Memory tile の L1.5/L2/DDR split は、tile 単体の property ではありません。同じ load-tile coordinate でも、swizzle、wave occupancy、どの neighboring tile が tensor data を共有するかに応じて cache に hit する場合も DDR へ抜ける場合もあります。GEMM の $M$-axis tile 間で B-tile reuse を維持すると DDR traffic を約 ${\sim}4\times$ 削減でき、私たちの motivating case では block swizzling により L2 hit rate が 35% から 72% に変化します。さらに現代の GPU は中間 L1.5/LRC tier (H200、B200) を追加しているため、単一の flat bandwidth term では不十分です。Reuse-distance analysis は cache modeling のために確立されていますが [Lam91], [Con98], [Nug14], [Ara19], [Ara20], [Niu12]、従来の formulation は cache-line trace を対象とし、analytical schedule search に組み込むには low-level すぎます。そこで TileSight は reuse distance を tile execution plan へ引き上げ、symbolic tile order を分析対象 sequence、tile-sized tensor block を reuse universe とします。私たちの知る限り、tile-granular reuse-distance abstraction により schedule-sensitive な multi-level cache modeling を analytical GPU performance model 内で実用可能にした初の手法です。

#### 3.5.1 Tensor Access と Tile Reuse Distance

TileSight は、tile grid に関連する各 tensor について *tensor access* を導入します。これは tile ごとの footprint、placement descriptor、repeated-access count、同じ data block が reuse される grid dimension からなります。Reuse dimension `reuse_dims` により、1 つの rule で多様な operator を扱えます。Tensor の reuse key は、tile coordinate を non-reuse dimension に射影したものです。GEMM grid $(M_t,N_t)$ では、A tile は $N_t$ 全体、B tile は $M_t$ 全体で reuse されます。MLA decode では、key-value (KV)-cache tile は同じ batch element の attention head 間で reuse されます。Convolution では、weight と activation は batch、output-channel、spatial axis に関して異なる reuse dimension を持ちます。これにより、reuse を決定する schedule information を維持しつつ、operator-specific cache formula を避けられます。

*Tile reuse distance* $D_T$ は、同じ tensor block への 2 回の連続 access の間に access される、異なる tile-sized data block の数です。従来の reuse distance は、2 回の access の間にいくつの cache line または memory transaction が介在するかを問います。tile reuse distance は、GPU kernel schedule が公開する unit で同じ問いを立てます。128-byte cache line の代わりに 8 KB tile を model 化すると、追跡する entry は $64\times$ 少なくなり、tile-centric schedule が公開する granularity と一致します。また、block swizzle と traversal order が cache model に直接現れ、trace-level cache simulation を回避できます。

<span id="figure-04"></span>

![図 4: Tile reuse distance と cache-line reuse distance。左: 従来の cache-line reuse distance は数万の line entry を追跡し、line granularity で exact SDCM を評価します。右: TileSight は reuse distance を tile-sized block へ引き上げ、Gaussian SDCM approximation を適用し、reduction axis に沿って sampling することで、schedule sensitivity を維持しながら cache modeling を軽量化します。](../../papers/tilesight/figure-04.png)

`reuse_dims` を持つ tensor について、TileSight は tile の non-reuse coordinate から reuse key を計算します。

$$
\mathrm{key}(\mathbf{x}, R)=\mathrm{Linearize}\bigl(x_d\mid d\notin R\bigr),
\tag{6}
$$

ここで $\mathbf{x}$ は tile coordinate、$R$ は reuse dimension の集合です。GEMM の A matrix について $R=\{N_t\}$ であり、同じ M-row の全 tile が同じ A key を共有します。B については、同じ N-column の全 tile が同じ B key を共有します。Swizzle と row-panel traversal を含む具体的 tile execution order が、これらの key が現れる sequence、したがって reuse distance を決めます。

#### 3.5.2 Hit Probability と高速評価

Reuse distance $D_T$、associativity $A$、tile 単位の cache capacity $B_T$ が与えられると、stochastic distance cache model は least-recently-used (LRU)-like cache の hit probability を推定します。Exact SDCM hit probability は binomial form で表せます。

$$
P(h \mid D_T) =
\sum_{a=0}^{A-1}
\binom{D_T}{a}
\left(\frac{A}{B_T}\right)^a
\left(\frac{B_T-A}{B_T}\right)^{D_T-a},
\tag{7}
$$

ここで $A$ は cache associativity、$B_T$ は tile 単位の cache capacity です。この binomial form は正確ですが、大規模 tile grid の各 tensor key について計算すると高 cost です。

そこで TileSight は効率的評価のため Gaussian approximation を採用します。

$$
P(h \mid D_T)_{\mathrm{approx}}
=
1 - Q\!\left(
\frac{|A-1-\mu|}{\sqrt{\sigma^2}}
\right),
\tag{8}
$$

ここで

$$
\mu = D_T \cdot \frac{A}{B_T},
\qquad
\sigma^2 =
D_T \cdot \frac{A}{B_T}
\cdot
\left(1-\frac{A}{B_T}\right).
\tag{9}
$$

$Q(x)$ は standard normal distribution の complementary cumulative distribution function (CDF) です。Overhead をさらに減らすため、CDF $\Phi(x)$ に Zelen-Severo approximation [Abr65] を適用します。

$$
\Phi(x)
\approx
1 -
\left(a_1t-a_2t^2+a_3t^3\right)
\frac{e^{-x^2/2}}{\sqrt{2\pi}},
\tag{10}
$$

ここで $t=(1+0.33267x)^{-1}$、$a_1,a_2,a_3$ は constant です。

**Reduction axis に沿った sampling。** Tile execution plan は reduction axis (GEMM の $K$ axis など) を公開します。TileSight は inner-loop access をすべて replay するのではなく、この granularity で reuse event を sample します ($K{=}8192$、$\mathrm{tile}_K{=}32$ の GEMM では accuracy loss をほぼ生じさせずに check を $256\times$ 削減)。Tile-level reuse distance と Gaussian approximation を組み合わせることで、cache-model evaluation を約 5 桁削減し、offline trace analysis ではなく analytical loop 内で cache modeling を実行可能にします。

#### 3.5.3 Two-Level Cascade、Swizzle、Wave

中間 L1.5/LRC tier を持つ GPU では、TileSight は SDCM を cascade として適用します。各 physical SM group 内では L1.5、global には L2 を用い、DDR が残りの miss traffic を担います。この設計がなければ L1.5 hit probability は 0 となり、model は単一 L2 evaluation に縮退します。Block swizzle、row-panel、Z-order、persistent-block schedule は、reuse-distance simulation に渡す具体的 tile coordinate sequence にすぎません。Wave 内では、TileSight は hardware nondeterminism、sequential tensor load、cross-tensor cache aging に応じて $D_T$ を perturb します。これらはすべて tile execution plan と hardware grouping から導出され、kernel-specific profiling を必要としません。Tail wave は SM の一部を使用するため shared bandwidth のより大きな share を受け、tail について envelope が再計算されます。得られた L1.5/L2/DDR byte count が式 1 の対応 entry を埋めるため、cache behavior は final latency だけでなく pipeline envelope 自体を変えます。

### 3.6 Device 間 Tile

Device 間実行は、同じ tile 内 abstraction の placement extension です。Tile の source または destination は別 GPU 上の shard/replica を指すことができ、その resource vector は non-zero の `Net` entry を得ます。Tensor、expert、sequence、data-parallel mapping は tile grid とその tensor tile の両方を partition し、GPU group 全体の placement descriptor を生成します。Partition 後、local tile wave は、別 device が生成した tensor tile、replicated activation、または後続 tile が consume する前に reduce すべき partial result を必要とする場合があります。TileSight はこれらを *remote tensor access* として扱います。必要な collective または point-to-point transfer は producer-consumer placement から直接推定され、それぞれが source/destination device、byte volume、`Net` resource usage を持つ tile になります。

**Logical exchange と topology。** 推定された各 remote tensor access について、TileSight は必要な tensor-tile movement を ordered stage に分解します。Stage は logical source-destination exchange $(s,d,b)$ で表されます。$s$ は tensor tile を所有または生成する device、$d$ はその tile wave が tensor tile を consume する device、$b$ は tensor access から導出される tile/shard byte volume です。Collective algorithm は異なる stage decomposition を提供するだけです。ring all-reduce は reduce-scatter step と all-gather step、tree algorithm は reduction level と broadcast level を用い、irregular routing は point-to-point のままです。この representation は packet-level ではなく tile-level です。Communication volume の推論に必要な tensor-placement information を維持しつつ、各 exchange をどの physical network-on-chip (NoC) または interconnect link が運ぶかは hardware topology に委ねます。

**Stage ごとの routed cost。** Stage 内の exchange を routing した後、TileSight は hop latency と bottleneck-link serialization への分解に対応する $\alpha$-$\beta$ communication model [Tha05] によって stage time を推定します。

$$
T_k
=
\underbrace{
\max_{(s,d,b)\in\mathcal{E}_k}
\sum_{l\in\mathcal{P}_{sd}} \alpha_l
}_{\mathrm{routed\ hop\ latency}}
+
\underbrace{
\max_{l\in\mathcal{L}} \beta_l B_{l,k}
}_{\mathrm{bottleneck\ link\ serialization}},
\tag{11}
$$

ここで $\mathcal{E}_k$ は stage $k$ の logical exchange 集合、$\mathcal{P}_{sd}$ は $(s,d,b)$ の physical route、$B_{l,k}$ は link $l$ を通って routing される byte、$\alpha_l,\beta_l$ は link $l$ の calibration 済み startup latency と inverse bandwidth です。推定された communication sequence の cost は stage の ordered sum、$T_c=\sum_{k\in\mathcal{K}_c}T_k$ です。Ring collective のように同一 stage を反復する algorithm では、TileSight は 1 stage を評価して stage count を乗じます。結果は式 1 の `Net` dimension に入り、device 間 movement は tile 内 resource requirement として表現され、他のすべてと同じ steady-state machinery により local compute と overlap します。

### 3.7 全体の統合

各要素が揃うと、Algorithm 1 の完全な意味が明らかになります。Cache analysis (第 3.5 節) は tile 間 execution order から $\mathbf{u}(o)$ の L1.5/L2/DDR entry を埋めます。Remote tensor access (第 3.6 節) は routed $\alpha$-$\beta$ stage cost から `Net` entry を埋めます。Envelope (第 3.4 節) は、完成した resource vector と dependency/concurrency edge (第 3.3 節) を消費し、nested loop、wave、network stage に再帰的に適用されます。これらは post-hoc correction ではありません。各要素は、envelope を流れる同一の tile ごとの resource vector を埋めるか消費します。

### 3.8 可搬な Hardware Abstraction

TileSight が必要とするのは、tile execution plan とその placement descriptor に影響する parameter だけです。この abstraction は tensor-placement hierarchy に対応します。Local placement は register/TMEM/SMEM/cache/DDR resource に、remote placement は GPU/node 間の calibration 済み network hierarchy に対応します ([表 3](#table-03))。値は vendor specification と、practical bandwidth、utilization cap、network parameter のための軽量 microbenchmark から得られます。

**表 3: 本稿で評価する GPU architecture の hardware specification。theoretical peak (spec) / microbenchmark-calibrated (meas.)。**

<span id="table-03"></span>

| GPU | SM 数 | VEC FP32 T spec / meas. | TC FP16 T spec / meas. | SFU T spec / meas. | L2 TB/s meas. | DDR TB/s spec / meas. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A100 | 108 | 19.5 / 19.0 | 312 / 299 | 2.4 / 2.4 | 3.2 | 1.9 / 1.7 |
| H200\* | 132 | 61.8 / 49.5 | 989 / 928 | 3.9 / 4.1 | 9.2 | 4.8 / 4.2 |
| B6000 | 188 | 117 / 88.6 | 468 / 433 | 7.3 / 6.7 | 7.6 | 1.8 / 1.4 |
| B200 | 148 | 74.5 / 57.7 | 2382 / 2185 | 4.7 / 4.5 | 20.5 | 8.0 / 7.0 |
| MI210 | 104 | 45.3 / 34.4 | 181 / 167 | 2.8 / 1.1 | 4.8 | 1.6 / 1.4 |

*注*: TileSight の hardware abstraction には cache hierarchy、architecture-specific TMEM bandwidth、SMEM/occupancy limit、GPU group 間の network hierarchy も含まれます。簡潔さのため、ここには記載していません。\*H200 の maximum clock は 1980 MHz、default clock は 1830 MHz です。

TileSight は、warp-level instruction issue、compiler register allocation、instruction granularity の hardware scheduling、packet-level network effect を model 化しません。代わりに、tile-level programmer と distributed runtime が制御する schedule-visible effect、すなわち tile shape、tensor placement、reuse pattern、swizzle order、pipeline depth、SM あたりの resident block、distributed partitioning、collective algorithm、topology-aware routing を model 化します。これにより、model は GPU generation 間で portable であり、schedule search 内で使用できるほど高速になります。

## 4 実装

TileSight は Python (約 6K lines) で実装され、NVIDIA GPU と AMD GPU をサポートします。ユーザーは kernel を tile-based program として記述します。Triton/TileLang code から抽出することも、non-DSL kernel について手書きすることもでき、TileSight は kernel を実行せずに完全な performance breakdown を生成します。

**任意の fused program の記述。** 任意の kernel を表現するため、TileSight は各 tile 内で実行される operation を tile-action DAG として記述します (第 3.1 節)。各 tile action は `HardwareUsage` resource vector (第 3.1 節) と 2 つの追加 attribute で注釈されます。(1) action 間の明示的 data dependency。(2) *intermediate result が存在する scratchpad memory level*、すなわち register file、shared memory、または Blackwell の architecture-specific tensor memory (TMEM)。Scratchpad annotation は、action 間の各 data movement に課される bandwidth tier と、消費される on-chip capacity を決めます。後者は occupancy を制約します。Data dependency は tile-action node 間で宣言されます。TileSight は dependency と整合するすべての valid topological ordering を自動列挙し、tile latency を最小化する schedule を選択します。

**Software pipeline と occupancy。** Pipelined kernel について、ユーザーは Triton の `num_stages` または TileLang の明示的 stage count に対応する pipeline depth を与えます。Tile あたり shared memory や register count などの kernel resource usage が与えられると、TileSight は resource-limited minimum として SM あたり resident tile 数を計算します。これが effective pipeline depth と per-SM bandwidth allocation を決めます。TileSight は head wave と tail wave を別々に model 化します。Tail wave は active SM が少ないため、各 SM が L2/DDR bandwidth のより大きな per-SM share を持ち、これが tile ごとの latency calculation に反映されます。

**Single GPU から cluster へ。** Single-GPU level では、tile grid 全体を 1 device に schedule します。Node level では、`DistributedTileMap` が grid を GPU 間で partition し、`NetworkHierarchy` が NVLink や PCIe を含む intra-node interconnect を捉えます。TileSight は message size と device count に基づいて ring、recursive-doubling、Rabenseifner などの collective algorithm を選択します。Multi-node cluster では、同じ `NetworkHierarchy` に InfiniBand や NVLink Bridge などの inter-node link を追加します。ユーザーは任意 link の per-hop bandwidth と latency を指定して custom topology を定義できます。`DistributedTileMap` が与えられると、TileSight は partition された tile grid の producer-consumer placement から必要な remote tensor access を推定し、それぞれを $(s,d,b)$ logical exchange の ordered stage に分解し、`NetworkHierarchy` 上で $\alpha$-$\beta$ stage cost を適用します。これにより tile ごとの `Net` resource time が生成され、local compute/memory と同じ pipeline envelope を流れます。

<span id="figure-05"></span>

![図 5: A100、B200、B6000、H200、MI210 における GEMM latency prediction と measured latency の比較。各点は 1 つの BF16/FP16 tensor-core GEMM shape を表し、対角線は完全な予測を示します。](../../papers/tilesight/figure-05.png)

**表 4: H100 上の FlashAttention-3 modeling と NCU の比較 (Qwen configuration: batch 1、64 heads、head-dim 128)。NCU が ground truth。**

<span id="table-04"></span>

|  | Time (ms) | L2 hit (%) | L2 util. (%) | SMEM (%) | TC (%) | SFU (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NCU | 5.58 | 96.50 | 38.66 | 51.14 | 74.78 | 38.58 |
| TileSight | 5.73 | 95.26 | 35.72 | 43.13 | 70.30 | 35.42 |

**Composition と calibration。** Modeling chain は bottom-up に動作します。Cache model は tile schedule と reuse distance に基づいて L1.5/L2/DDR traffic fraction を計算します。この traffic data が tile ごとの pipeline overlap model に入力されます。Wave model は tile ごとの結果を device ごとの time へ aggregate し、distributed model は communication を加えて overlap を計算します。Memory/TMEM bandwidth、unit ごとの compute throughput、その他 hardware parameter は、小規模 microbenchmark によって architecture ごとに 1 回 calibration されます。これには[図 2](#figure-02) のような working-set size 全体の bandwidth sweep と、数秒しか要しない短い matrix-multiply probe が含まれます。

<span id="figure-06"></span>

![図 6: 4,680 の GEMM persistent-kernel case における TileSight の L2 hit-rate prediction と NCU ground truth の比較。](../../papers/tilesight/figure-06.png)

## 5 評価

<span id="figure-07"></span>

![図 7: H200 $\times 8$ と B200 $\times 8$ における AllGather、AllReduce、ReduceScatter、All-to-All の pure-collective prediction。](../../papers/tilesight/figure-07.png)

統一された tile-level abstraction により、target ごとの retraining や profiling を行わずに single-operator latency、cache behavior、distributed kernel、end-to-end serving を model 化できることを示すため、A100、H200、B200、B6000、H200-NVL、B200 $\times 32$ system において、single-GPU kernel から multi-GPU LLM serving まで TileSight を評価します。

まず第 5.1 節で、hardware/framework configuration、workload、baseline を含む experimental setup を説明します。次に第 5.2 節で single-GPU operator によって core tile-level model を検証し、第 5.3 節で persistent kernel の L2 cache prediction を詳しく分析します。第 5.4 節と第 5.5 節では evaluation を distributed setting へ拡張し、collective/fused compute-communication kernel と end-to-end vLLM serving の両方を扱います。最後に第 5.6 節で、TileSight を performance diagnosis と cost-model-guided schedule pruning に利用する方法を示します。

### 5.1 実験設定

**Hardware と framework。** 幅広い hardware coverage を確保するため、A100 $\times 1$、H200-SXM $\times 8$、B200 $\times 8$、InfiniBand 接続 B200 $\times 32$ cluster、B6000 $\times 2$、H200-NVL $\times 8$ system を評価し、SXM、PCIe、NVLink4/5、multi-node setting を網羅します。Hopper/Amphere machine では CUDA 12.9、Blackwell machine では CUDA 13.1 を使用します。Cross-vendor coverage のため、ROCm 6.2 上の AMD MI210 (CDNA2) も評価します。GEMM measurement は NVIDIA GPU で `cutlass_profiler`、MI210 で Composable Kernel (CK) を使用します。Distributed kernel は Parallel Kittens [Sul25]、end-to-end serving は vLLM 0.19.0 を使用します。

**Workload。** Kernel-level experiment は BF16/FP16 GEMM、persistent-kernel cache sweep、collective、fused compute-communication kernel を対象とします。End-to-end vLLM experiment には、Qwen、Llama、DeepSeek family の dense model と MoE model が含まれ、single-GPU serving から最大 32 GPU の tensor/expert/data-parallel serving までを扱います。合計で 703 GEMM shape、cache modeling 用の 4,680 persistent-kernel case、166 vLLM decode configuration を評価します。

**Baseline。** Single-operator prediction では、Roofline [Wil09]、NeuSight [Lee25][+1]、PipeWeave [Zha26p]、GenZ [Bam24] と比較します。NeuSight は A100 を含む 6 種類の PipeWeave GPU から得た BF16/FP16 GEMM data で training します。PipeWeave dataset は A100 と Hopper-class machine を含むため、それらの architecture において PipeWeave は zero-shot baseline ではありません。Distributed kernel と end-to-end serving では、PipeWeave と GenZ と比較します。PipeWeave の collective model は GPU ごとの random forest であり、設定可能な $\alpha$-$\beta$ parameter や topology parameter を持ちません。私たちの target のうち A100 と B6000 (RTX PRO 6000 Blackwell) には native PipeWeave collective dataset がありますが、H200-NVL と B200 は非対応であるため、最も近い代替として H800 dataset を使用します。End-to-end serving については、必要な vendor hardware specification を PipeWeave に与えます。

[+1]: 元の NeuSight は FP32 GEMM のみで training されています。公平な比較のため、PipeWeave の FP16 dataset で retraining します。

### 5.2 Single-Operator Prediction Accuracy

<span id="figure-08"></span>

![図 8: H200 $\times 8$ と B200 $\times 8$ における fused compute-communication kernel prediction (AllGather+GEMM、GEMM+ReduceScatter、Ulysses Attention)。](../../papers/tilesight/figure-08.png)

<span id="figure-09"></span>

![図 9: Dense LLM、MoE model、multi-node configuration にわたる vLLM decode throughput prediction。Dense row は A100 $\times 1$、B6000 $\times 2$、B200 $\times 8$、H200-NVL を、MoE row は B200 $\times 8$、B200 $\times 32$、H200-NVL $\times 8$ を対象とします。Bar は measured vLLM tokens per second と TileSight、および対応する場合の PipeWeave を比較します。PipeWeave は MoE に対応しません。](../../papers/tilesight/figure-09.png)

<span id="figure-10"></span>

![図 10: すべての healthy configuration における predicted decode throughput と measured decode throughput。TileSight: overall 13.52% wMAPE。PipeWeave: 対応する dense row で 31.84% wMAPE。](../../papers/tilesight/figure-10.png)

[図 5](#figure-05) では、A100、B200、B6000、H200 上の 703 BF16/FP16 tensor-core GEMM shape を評価し、stream-K path と single-instruction, multiple-thread (SIMT) fallback path を除外した `cutlass_profiler` measurement を ground truth とします。TileSight の pooled MAPE は 12.35% であり、PipeWeave の 21.97%、retraining した NeuSight の 32.95%、Roofline の 33.85%、GenZ の 34.89% と比較して優れています。TileSight は、より新しい B200、B6000、H200 target で最良です。A100 は NeuSight の training distribution に含まれるため、NeuSight が A100 では僅差で上回りますが、この優位性は新しい GPU へ転移しません。これは architecture-specific learned predictor の overfitting risk を示しています。MI210 では、CK が `cutlass_profiler` のような明示的 rasterization (along-$M$/along-$N$) または swizzle control を提供しないため、TileSight は default cache mode で動作します。それでも 23.4% MAPE で首位となり、PipeWeave (25.5%)、NeuSight (26.4%)、Roofline (38.8%)、GenZ (40.4%) を上回ります。Non-GEMM fused operator は、以下の distributed workload と end-to-end workload で評価します。

[表 4](#table-04) は fused FA-3 kernel で TileSight と NCU を比較します。最終 model は latency を 2.7% 以内で予測し、主要な resource-utilization component を追跡します。これは non-GEMM fused execution における tile-pipeline model の簡潔な sanity check です。

**表 5: TileSight が診断した kernel の performance improvement。**

<span id="table-05"></span>

| Kernel | Framework | Device | Baseline | 問題 | 解決策 | 最適化後 | Speedup |
| --- | --- | --- | ---: | --- | --- | ---: | ---: |
| ReLU | Triton | MI210 | 1.40 ms | indirect addressing | address unroll | 1.10 ms | $1.27\times$ |
| Avg_Pool | Triton | MI210 | 0.20 ms | indirect addressing + overlap なし | address unroll + small tile | 0.10 ms | $2.00\times$ |
| Avg_Pool | Torch | MI210 | 0.15 ms | overlap なし | small tile | 0.10 ms | $1.50\times$ |
| GEMM(M128) | CK | MI210 | 3.68 ms | overlap なし | SM あたり複数 thread block | 2.68 ms | $1.37\times$ |
| GEMM(K57344) | CK | MI210 | 55.63 ms | large K による L2 hit-rate 問題 | large tilek -> CU あたり 1 TB | 51.90 ms | $1.07\times$ |
| RMS_Norm | Torch.Compile | H100 | 0.21 ms | overlap なし | SM あたり複数 thread block | 0.18 ms | $1.17\times$ |
| MLA(kv8192 b128 h128) | Triton | MI210 | 66.38 ms | tiling、memory allocation、SMEM conflict | register allocation、larger tile、conflict elimination | 7.40 ms | **$8.97\times$** |

<span id="figure-11"></span>

![図 11: TileSight が exhaustive autotuning に代わって Triton と TileLang の tile configuration selection を導く場合の H100/MI210 上の kernel performance。Reference line は multi-head attention/grouped-query attention (MHA/GQA) の FlashAttention-3、MLA の FlashMLA、matrix multiplication の cuBLAS/rocBLAS、dequantized matrix multiplication の vendor library です。](../../papers/tilesight/figure-11.png)

<span id="figure-12"></span>

![図 12: TileLang の cost model としての TileSight。candidate schedule の 95% を prune し、予測上位 5% を残すことで、LLaMA 由来の 10 GEMM-FP16 workload において exhaustive-search best performance の平均 99.66% を達成します。](../../papers/tilesight/figure-12.png)

### 5.3 L2 Cache Prediction Accuracy

[図 6](#figure-06) では、4,680 GEMM persistent-kernel case において tile reuse-distance cache modeling を NCU と比較します。[図 2](#figure-02) の bandwidth sweep で calibration した effective cache capacity を用いると、mean absolute L2 hit-rate error は各 GPU で約 1 percentage point にとどまります。A100 で 1.46 pp、H200 で 0.88 pp、B200 で 1.05 pp、B6000 で 0.78 pp です。結果は tile reuse-distance cache modeling の有効性を示します。

**SM 間 execution skew の影響。** Reuse-distance model は tile が均一な速度で進むと仮定しますが、SM は同期を失い、同時に異なる $K$-slice を処理します。そのため同時 access される tile は L2 を越えて広がり、deep-$K$ GEMM では measured hit rate が TileSight の lockstep prediction より低くなります (たとえば H200 上の $M{=}N{=}8192$、$K{=}28672$ の GEMM では、予測 82% に対して測定 43%)。このような configuration は稀であるため aggregate error は約 1 percentage point にとどまりますが、この regime では model が系統的に楽観的です。第 7 節で再び取り上げます。

### 5.4 Distributed Validation

[図 7](#figure-07) と[図 8](#figure-08) は H200 $\times 8$ と B200 $\times 8$ 上の 304 distributed case、すなわち 152 pure collective と 152 fused compute-communication kernel を検証します。TileSight は logical source-destination exchange を抽出し、calibration 済み NVLink topology 上で routing し、第 3.6 節の $\alpha$-$\beta$ model で各 stage を評価します。Pure collective では TileSight は 12.22% wMAPE を達成し、GenZ は 20.82%、対応する row における PipeWeave は 65.72% です。PipeWeave はこれらの collective に対応する native configurable H200/B200 backend を持たず、H800 random-forest model へ fallback するため、私たちの machine における NVLink4/5 bandwidth の違いを反映できません。B200 Ulysses Attention では、local compute stage は source-aligned SM100 $128\!\times\!128$ FA4 tile pipeline を用い、TMEM traffic、packed grid、sectioned LPT mapping と 4 つの all-to-all stage を合成します。両 baseline が対応しない fused kernel で、TileSight は 14.83% wMAPE を達成します。

### 5.5 vLLM End-to-End Decode

[図 9](#figure-09) と[図 10](#figure-10) は、dense、MoE、single-node、multi-node serving にまたがる 166 healthy configuration で end-to-end vLLM decode throughput を評価します。評価 system は A100 $\times 1$ と B6000 $\times 2$ から B200 $\times 32$ と H200-NVL $\times 8$ まで広がり、local tile execution と routed distributed stage の両方を実行します。全体として TileSight は 13.52% wMAPE を達成し、B200 extension を備えた PipeWeave は 114/117 dense configuration で 31.84% wMAPE です。PipeWeave は MoE に対応しません。PipeWeave は A100 と B6000 で native collective dataset を使用しますが、H200-NVL と B200 では H800 に fallback します。B200 については B200 hardware specification を与えて PipeWeave を拡張し、GEMM-configuration lookup には最も近い H800 sample、calculator には Hopper calculator を使用します。B200 extension は 19/22 dense configuration で valid prediction を生成します。残る 3 つの large-batch case では、prefill RMSNorm sequence length が PipeWeave の 131K-token MLP training maximum を超えます。PipeWeave は sigmoid により learned utilization factor を $[0,1]$ に制限していますが、これらの out-of-range input は factor を 0 にして division by zero を引き起こし、この case に対する robust end-to-end prediction を妨げます。これは、未見 case へ extrapolate する際の ML-based predictor の robustness limitation を示します。TileSight は machine ごとに 7.5-18.0% wMAPE、MoE configuration で 10.35% wMAPE を達成します。

### 5.6 主な応用: Diagnosis と Cost Model

Interpretable な性質により、TileSight は white-box optimization aid として使用できます。[図 11](#figure-11) は、TileSight が選択した tile configuration が H100 と MI210 上の attention、MLA、GEMM、dequantized matmul kernel において、強力な vendor baseline と expert baseline に匹敵またはそれを上回ることを示します。[図 12](#figure-12) は、同じ model を TileLang cost model として使用した結果です。予測上位 5% の schedule を保持すると candidate の 95% を prune しつつ、exhaustive-search best performance の平均 99.66% に到達します。これは、learned または vendor-tuned cost model が弱い guidance しか提供できない、support の少ない target で特に有用です。Analytical model はその場合も高品質な schedule candidate を見つけられます。

Diagnosis case は、indirect addressing、不十分な pipeline overlap、poor L2 locality、architecture-specific memory-layout issue という 4 つの繰り返し現れる bottleneck class に分類されます。TileSight は各 case で、bottleneck を address unrolling、tile-size adjustment、より高い resident-block occupancy、shared-memory/register-layout fix などの具体的 tile-level change に対応付けます。[表 5](#table-05) は TileSight が indirect addressing、pipeline stall、L2 locality bottleneck を特定し、$1.07$-$8.97\times$ の improvement につながった diagnosis case をまとめています。

## 6 関連研究

**Tile 中心プログラミング フレームワーク。** Triton [Til19]、TileLang [Til25g]、TileLink [Zhe25t]、CUTLASS/CUTE [Nvi24]、CuteDSL [NviCut]、ThunderKittens [Ben25]、FractalTensor [Liu24t]、NVIDIA CUDA Tile [Nvi26] は、GPU programming を tile-centric abstraction へと進めてきました。しかし、tile-centric performance model を備えるものはありません。Triton は black-box autotuning、TileLang は heuristic、tritonBLAS [Swa25] は GEMM-specific analytical selection に依存します。TileSight は、この framework 群のための統一 tile-centric cost model および diagnosis backend として、この隔たりを埋めます。

**Performance Modeling と Prediction。** Roofline [Wil09] とその variant (GenZ [Bam24]、[Mor24]、[Yua24t]、[Pat25t]、[Dav25] など) は LLM inference に有用な first-order bound を提供しますが、FLOP/byte count が同一で schedule が異なる kernel を区別できず、異なる tile order における L2 reuse などの schedule-dependent effect を捉えられません。Karami ら [Kar25] はさらに、non-GEMM operation が inference latency の最大 74% を占めることを示し、GEMM-centric assumption に異議を唱えています。Dataflow exploration framework [Par19]、[Gao19t]、[Kwo20]、[Zhe23]、[Wu22]、[Cai23] は spatial accelerator の loop nest と data reuse を model 化しますが、simplified hardware assumption に依存するため GPU への適用が制限されます。Hybrid approach と ML-based approach、すなわち PipeWeave [Zha26p]、NeuSight [Lee25]、CDMPP [Hu24t]、TAO [Pan24]、Omniwise [Wan25o]、その他 [Geo21]、[Li23t] は、learned model によって runtime を予測します (end-to-end、または analytical estimate 上の residual として)。多くの場合は正確ですが、kernel が*なぜ*遅いのかを示さない black box です。TileSight は learned component を持たない完全な first-principles model でありながら、これらの predictor と同等以上の accuracy を実現し、performance を actionable component へ分解する schedule-aware、tile-granular diagnosis を提供する点で異なります。

**GPU Profiling と Instrumentation。** Vendor profiler (Nsight Compute [Nvi25n]、OmniPerf [AMD25]) は metric を報告しますが、root-cause guidance はほとんどありません。KPerfIR [Gua25] と Neutrino [Hua25] は compiler-based/probe-based GPU instrumentation を発展させ、binary-level tool [She18]、[Zho21a]、[Zho21b]、[Zen24] は low-level visibility を提供します。すべて*事後的*であり、execution を必要とし、未見 configuration を予測できません。TileSight は execution 前に performance を予測し、bottleneck を tile-level scheduling decision に対応付けます。

**Distributed Multi-GPU Performance Modeling。** Vidur [Agr24]、Lumos [Lia25]、SimAI [Wan25s]、TokenSim [Wu25t]、Maya [Yar25]、Echo [Fen24] は profiling-based kernel estimator を用いて大規模 distributed training/inference を simulate します。一方、DistServe [Zho24]、CrossPipe [Che25t]、Sailor [Str25]、Metis [Um24]、RAPID-LLM [Kar25a] はさまざまな communication model と scheduling model によって parallel strategy を最適化します。いずれも single-GPU kernel execution を black box として扱います。TileSight は相補的な intra-kernel level で動作し、これらの distributed simulator に接続できる white-box tile-level cost estimation を提供します。同時に独自の distributed extension は、統一 tile abstraction の下で tile-level prediction と communication model を合成します。

## 7 制約と今後の課題

TileSight は、runtime が resource utilization に支配される regular tile-structured program を対象とします。Data-dependent control flow、高度に irregular な memory access、instruction-level compiler decision、非公開の warp/cooperative thread array (CTA) scheduling、closed-source runtime behavior は model 化しません。Hardware abstraction は throughput に重点を置きます。Small-batch decode attention のような latency-bound case と、B200 SM-to-HBM affinity のような multi-die effect には、より細粒度の latency parameter と topology parameter が必要です。また TileSight は、tile が SM 間で均一な速度で実行されると仮定します。実際には SM が同期を失うため、large-$K$ GEMM における L2 hit-rate prediction はやや楽観的になります (第 5.3 節)。最後に、TileSight は選択した GEMM workload で TileLang cost model として検証されていますが、より広範な compiler integration と non-GEMM schedule search は今後の課題です。

## 8 結論

TileSight は、Triton、TileLang、CUDA Tile、CuteDSL に共通する GPU programming unit となった tile が、performance reasoning も統一できることを示します。Resource usage、dependency、cache reuse、device 間 placement を tile level で model 化することにより、TileSight は architecture ごとの training や profiling を行わず、single kernel から multi-node cluster まで performance を正確に予測します。より広い教訓は first-principles にあります。物理的根拠を持つ少数の mechanism から始め、その composition に complex execution を説明させることです。Regular tile-structured workload では、この approach によって architecture 間を transfer する正確で interpretable な prediction が可能です。
