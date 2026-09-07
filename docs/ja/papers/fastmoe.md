---
title: 'FastMoE: A Fast Mixture-of-Expert Training System'
createTime: 2026/09/07 21:06:24
permalink: /ja/papers/fastmoe/
---

> [Jiaao He](https://dblp.org/pid/249/2660)、[Jiezhong Qiu](https://jiezhongqiu.com/)、[Aohan Zeng](https://blog.sengxian.com/)、[Zhilin Yang](https://kimiyoung.github.io/)、[Jidong Zhai](https://pacman.cs.tsinghua.edu.cn/~zjd/)、[Jie Tang](https://keg.cs.tsinghua.edu.cn/persons/jietang/)。2021 年 3 月 24 日に arXiv へ初回投稿、現行版は v1。[FastMoE: A Fast Mixture-of-Expert Training System](https://arxiv.org/abs/2103.13262)。<a href="/paper/fastmoe.pdf" target="_blank" rel="noopener noreferrer">原論文 PDF</a>。[DOI](https://doi.org/10.48550/arXiv.2103.13262)。[TeX ソース](https://export.arxiv.org/e-print/2103.13262)。正確な印刷レイアウトと参考文献については原論文 PDF を正とする。

## 概要

Mixture-of-Expert（MoE）は、言語モデルの規模を数兆パラメータまで拡大できる大きな可能性を持つ。しかし、数兆規模の MoE を訓練するには、十分に調整された高性能な分散訓練システムに向けて、アルゴリズムとシステムを協調設計する必要がある。残念ながら、この要件を満たす唯一の既存プラットフォームは Google のハードウェア（TPU）とソフトウェア（Mesh Tensorflow）スタックに強く依存し、一般には公開も提供もされておらず、とりわけ GPU および PyTorch コミュニティから利用できない。

本論文では、一般的なアクセラレータを用いる PyTorch ベースの分散 MoE 訓練システム *FastMoE* を提示する。このシステムは、柔軟なモデル設計と、Transformer-XL や Megatron-LM など異なるアプリケーションへの容易な適応を両立する階層的インターフェースを提供する。PyTorch で MoE モデルを直接実装する場合とは異なり、*FastMoE* では高度な高性能化技術によって訓練速度が大幅に最適化されている。このシステムは、複数ノードにまたがる複数の GPU に異なる expert を配置でき、GPU 数に対して expert 数を線形に増やせる。*FastMoE* のソースは Apache-2 ライセンスの下で [https://github.com/laekov/fastmoe](https://github.com/laekov/fastmoe) に公開されている。

<span id="section-1"></span>

## 1 はじめに

BERT [Dev18]、GPT-2/-3 [Rad19b, Bro20b]、XLNet [Yan19]、RoBERTa [Liu19a]、T5 [Raf20b]、GShard [Lep20]、Switch Transformer [Fed22] に代表される大規模言語モデルが近年登場したことで、自然言語処理研究の様相は劇的に変わり、GLUE [Wan18d] や SuperGLUE [Wan19h] など各種ベンチマークの新たな最先端ベースラインが改めて確立された。

考えられる多くの方法のうち、モデル規模の拡大は、より強力なモデルを実現する最も単純で効果的な方法の一つであることが示されている [Kap20]。$340$ million パラメータの BERT [Dev18] から、$11$ billion パラメータの T5 [Raf20b]、$175$ billion パラメータの GPT-3 [Bro20b] まで、モデル規模はわずか 2 年で $500\times$ に拡大した。さらに近年、GShard [Lep20] は記録的な $600$ billion パラメータまで拡大し、その記録は $1.6$ trillion パラメータの Switch Transformer [Fed22] によってすぐに更新された。GShard と Switch Transformer の巨大なモデル規模を支える主な要因は、mixture of experts（MoE）[Sha17] と呼ばれる新しいニューラルネットワークアーキテクチャである。

<span id="figure-01"></span>

![MoE 層の概略例](../../papers/fastmoe/figure-01.png)

**図 1。** MoE 層の概略例。この例では、gate が expert 1 と expert 3 を計算対象として選択する。

MoE 層（概略例は [図 1](#figure-01) を参照）は、gate と expert のプールから構成される。各入力について、gate が計算対象として選ぶ expert はごく少数に限られる。MoE の特殊なアーキテクチャは、大規模分散訓練にとって諸刃の剣である。一方では、expert を疎に活性化するため、計算量（FLOPs）を大幅に増やすことなく、モデル規模を桁違いに拡大できる。他方では、数千の expert まで拡張すると、MoE の不均衡な all-to-all 通信パターンが、アルゴリズムとシステムの協調設計に新たな課題をもたらす。したがって、PyTorch [Pas19] や TensorFlow [Aba16] など従来の深層学習ライブラリでは MoE を直接サポートできない。

新しいモデルアーキテクチャがもたらす課題により、研究コミュニティと産業界はいずれも、大規模分散訓練をサポートする MoE 実装を必要としている。しかし、PyTorch には素朴な単一 GPU 実装 [Rau19] がいくつか存在するものの、スケーラブルな MoE 訓練をサポートする現行唯一のシステムは、Google の非公開ハードウェアおよびソフトウェアスタック、すなわち TPU [Jou17a] と Mesh TensorFlow [Sha18a] を基盤としている。そのため、公開利用できるハードウェア（例：GPU）とプラットフォーム（例：PyTorch [Pas19]）上で MoE システムを開発することが急務である。

大規模 MoE 訓練のために、使いやすく、柔軟で、効率的かつスケーラブルなオープンソースの解決策を得たいという動機から、以下の設計目標を掲げて *FastMoE* を公開する。

- **使いやすさ：** MoE 層を定義するための使いやすいインターフェースを提供し、一般的な言語モデル訓練システム Megatron-LM [Sho19] をシームレスにサポートする。
- **柔軟性：** ユーザーが gate network と expert network を容易にカスタマイズできるようにする。
- **効率性：** Transformer 向けに高度に最適化された feedforward（FFN）層を統合する。
- **スケーラビリティ：** 複数ノードの複数 GPU にまたがる訓練によって、MoE モデルの規模拡大をサポートする。

従来の単一 GPU PyTorch 実装 [Rau19] と異なり、*FastMoE* は効率性とスケーラビリティを重視する。*FastMoE* には、専用最適化を施した高性能な専用 CUDA kernel が含まれる。*FastMoE* は NCCL [Jea17] を用いて、複数ノードの複数 GPU にまたがって動作できる。*FastMoE* は通信の詳細をモデル開発者から隠蔽する。*FastMoE* の model-parallel 手法では、expert を異なる GPU に分散しつつ、モデルの他の部分を batch 次元（data parallel）または tensor 次元（model parallel）で並列化したままにできる。expert 数に比例するモデル規模は、訓練に用いる GPU 数とともに拡大できる可能性があり、これが数兆規模のモデルを訓練する鍵となる。

実験では、単一 GPU 上で *FastMoE* が純粋な PyTorch API によるベースライン実装 [Rau19] より高速であることを確認した。Infiniband ネットワークで接続されたクラスタ上でノードをまたいで実行した場合にも、*FastMoE* は妥当なスケーラビリティを示す。分散 *FastMoE* を用い、各層に $96$ expert を持つ実際の GPT モデルを訓練したところ、良好な end-to-end 訓練速度が得られた。同じ計算量の非 MoE モデルと比べると、MoE アーキテクチャによってモデル規模が拡大したことが性能に寄与している。

本論文の構成は以下のとおりである。[第 2 節](#section-2) では MoE の背景を紹介し、既存システムを比較する。[第 3 節](#section-3) では *FastMoE* システムを詳しく提示する。[第 4 節](#section-4) では、高性能を達成する際の課題と *FastMoE* の解決策を紹介する。[第 5 節](#section-5) では、*FastMoE* の効率性と、訓練に *FastMoE* を用いた MoE モデルの性能向上を示す実験結果を提示する。[第 6 節](#section-6) では論文をまとめ、今後の研究の方向性を示す。

<span id="section-2"></span>

## 2 Mixture-of-Experts（MoE）

本節では、MoE のアーキテクチャと、MoE を訓練する現行システムを概観する。

<span id="section-2-1"></span>

### 2.1 MoE：モデル構造

Mixture-of-Expert は、[Sha17] が提案した Sparsely-Gated Mixture-of-Experts 層の略称である。MoE 層は複数の expert からなり、各 expert は任意のニューラルネットワークでよい。expert に課される唯一の制約は、同じ入力を受け取り、同じベクトル空間の出力を返すことである。[図 1](#figure-01) に MoE 層の詳しい例を示す。*gate network* と呼ばれる特殊なニューラルネットワークを導入し、与えられた入力に対して各 expert を評価する。このスコアに基づき、モデルごとに異なり得る方策で expert を選択する。次に、例の expert $1$ と $3$ のように、選択された expert を活性化して入力サンプルを処理する。expert の出力はスコアとともに、所定のアルゴリズムによって最終出力へ統合される。

一般的な expert 選択方法は、スコアが最も高い上位 $k$ 個の expert を選ぶことである。統合処理では、スコアを expert 出力の重みとして用い、各出力を全体の出力に加算する。勾配がスコアを通じて伝播できるため、gate network を訓練できる。[アルゴリズム 1](#algorithm-01) は以上の方法を形式化したものである。

<span id="algorithm-01"></span>

**アルゴリズム 1：top-$k$ gating を用いた MoE 層の forward 計算。**

- **要件：** $n$ 個の expert からなるプール：$\{E_1,E_2,\cdots,E_n\}$。
- **要件：** Gate $G$。
- **要件：** 選択する expert 数 $k$。
- **関数** $\operatorname{MoE}(x)$：
  - $\mathit{score}\leftarrow G(x)$。
  - $\mathit{indices}\leftarrow \operatorname{ArgMax}_k(\mathit{score})$。
  - $y\leftarrow$ $x$ と同形の zero tensor。
  - 各 index $i\in\mathit{indices}$ **について**：
    - $x_i\leftarrow E_i(x)$。
    - $y\leftarrow \mathit{score}_i*x_i+y$。
  - $y$ を**返す**。

<span id="section-2-2"></span>

### 2.2 現行の MoE 訓練システム

GShard システム [Lep20] は、MoE モデルの分散版を実装している。最大 $2048$ 個の TPU で言語モデルを訓練し、各 TPU には各層の expert を $1$ 個配置する。その結果、MoE 層は非 MoE 層の $2048\times$ のパラメータを含む。Switch Transformer [Fed22] では、モデルをさらに $1.6$ trillion まで拡大しており、このシステムが大規模なモデル訓練を強力にサポートできることが分かる。残念ながら、このシステムはまだ一般公開されていない。TPU クラスタに強く結び付いているため、汎用デバイス上で実験を再現することは難しい。さらに、GShard の設計は、異なる複製方策の下で expert の数と規模を変える柔軟性に欠ける。

Tensor2tensor [Vas18c] には MoE Transformer モデルが用意されている。しかし、この実装は GPU のサポートが十分でない Mesh TensorFlow [Sha18a] を使用する。Transformer で FFN を実装するには、複雑な `einsum` 演算子を含む TensorFlow コードが $100$ 行以上必要であり、開発者が構造を理解し、そのコードを基に別のモデル構造を探索する際の負担となる。

研究者の間で広く使われる深層学習フレームワーク PyTorch [Pas19] は、TensorFlow に比べて直接的なコーディング様式と柔軟性を備える。PyTorch で MoE モデルを訓練する試みも行われている [Rau19]。しかし、PyTorch コミュニティには多次元並列訓練ツールがなく、PyTorch ベースの実装はいずれも複数 GPU での訓練をサポートしない。MoE を導入する究極の目的はさらに大きなモデルを訓練することなので、PyTorch ベースの実装は候補にならない。

<span id="section-3"></span>

## 3 FastMoE：システム設計

本節では、分散訓練をサポートする *FastMoE* の設計を紹介する。

<span id="section-3-1"></span>

### 3.1 多様なモデル探索者のための柔軟なシステム

**任意の Expert Network を実行するための骨格。** *FastMoE* は、任意の network を expert として使用できる。*FastMoE* の `FMoE` インターフェースは任意のニューラルネットワークモジュールの constructor を入力として受け取り、そのモジュールを複数回複製して expert instance とする。expert は、整列された連続 input feature の batch を受け取り、同じ batch 順序で出力するよう定義される。したがって、expert module の実装は MoE アーキテクチャから分離され、開発者は自身の expert network の設計に集中できる。

さらに高い柔軟性を得るため、`FMoE` class は expert module が forward 計算を行う member function `expert_fn` を持つ。この関数は、MoE の動作をさらにカスタマイズするために overload できる。例えば、本節で後述する `FMoETransformerMLP` network では。expert のリストが、expert を並列に適用して latency を大幅に下げる専用最適化済み module に置き換えられる。

また、*FastMoE* は同一 worker 上への複数 expert の配置をサポートし、expert 数の設定空間をより柔軟にする（すなわち、expert 数は data parallel 数と等しくなくてもよい）。これは GShard の設計と異なる。

**Transformer 向けに高度に最適化された FFN。** MoE を用いた Transformer の訓練をより適切にサポートするため、*FastMoE* は標準的で高性能な FFN 実装（`FMoETransformerMLP`）を提供する。詳細な最適化方策は開発者から隠蔽される。

特に、同じ worker に複数の expert を配置する場合、素朴な実装では expert を loop して順番に forward を実行する。しかし、特定の種類の expert network では、並列実行による潜在的な高速化を利用できる。*FastMoE* では、主に専用の `FMoELinear` module によって fully-connected layer の並列実行を最適化する。専用に最適化された expert module は、expert module を順次計算する代わりに、利用可能な hardware resource の pool を維持し、expert の計算を並列に適用する。

**PyTorch と Megatron-LM のプラグイン形式サポート。** *FastMoE* は柔軟であるため、既存の訓練アプリケーションへ簡単に適応できる。Megatron-LM [Sho19] を例にすると、元の Megatron-LM モデルの FFN を MoE network へ迅速に置き換えるプラグイン形式の module が *FastMoE* に統合されている。listing 1 に示すように、この変換はわずか 2 行のコードで実現できる。

**Listing 1：Megatron-LM で FastMoE を使用するサンプルコード。**

```python
from fmoe.megatron import fmoefy
model = fmoefy(model, num_experts=<number of experts per worker>)
```

`fmoefy` 関数は Transformer 層内の FFN を見つけられる。次に *FastMoE* を用いる MoE network が作成される。これは interface level の互換性を保つため、`FMoETransformerMLP` module を wrap する module である。

<span id="section-3-2"></span>

### 3.2 分散方式によるモデル容量の拡大

**FastMoE の Model Parallel 手法。** モデル容量を拡大する最も効果的な方法の一つとして、多くの MoE モデルでは、大規模な expert 集団を収容して並列訓練する能力が求められる。モデル開発者にとって、GPU 間、さらにはノード間にわたる複雑なデータ転送を扱うことは難しい。高い訓練性能と良好な hardware resource 利用率を達成するには、computer architecture と parallel programming の専門知識が必要であり、一般的なモデル開発者の技術スタックを超えている。

*FastMoE* は、複数ノード上の複数 worker への expert の分散をサポートする。これは *FastMoE における model parallel 手法*と呼ばれる。input data 交換の詳細は `FMoE` インターフェース内に隠蔽される。モデル開発者は単一 expert のコードを書くだけでよく、各 expert には *FastMoE* が全 worker から集めたすべての input data が与えられる。その結果、モデル開発者は cross-worker 通信の実装詳細を考える必要がない。

*FastMoE* の設計では、worker 間で expert を分散する機能を有効にすると、forward 計算と backward 計算に追加の通信 operation が含まれる。operation を明確に区別するため、これらを global data exchange operation と呼び、[第 4 節](#section-4) で述べる local data shuffle process と対比する。

分散環境における主な課題の一つは、一つの worker 上にある全 expert へ割り当てられる input sample の総数が大きく変動し得ることである。gate output が得られる前に incoming sample 数を知ることはできない。しかし、input sample を格納する buffer の割り当ては、その数に依存する。そのため、worker 間で input sample を実際に交換する前に数量情報を交換し、expert count 情報の調査に従って memory を割り当てる。

<span id="figure-02"></span>

![global operation の例](../../papers/fastmoe/figure-02.png)

**図 2。** global operation の例。

[図 2](#figure-02) に *FastMoE* の global operation の例を示す。worker はまず、各 worker 上の各 expert に割り当てられた sample 数を数える。次に expert input の size を交換し、すべての worker が incoming input sample の数と、その送信元を取得する。各 receiving buffer の offset を計算すると、worker は data を直接交換し始める。incoming sample と outgoing sample の統計情報は、training iteration の処理全体を通して再利用できる点に注意が必要である。

**異種性を考慮した同期 Module。** network の異なる部分が異なる worker group に複製される可能性があるため、異種性が生じる。distributed module は、parameter の gradient を同期すべきか、また誰と同期すべきかを識別しなければならない。*FastMoE* はこの問題に対処するため、各 parameter に *data parallel communication group* tag を導入する。

tag は `world`、`data parallel`、`none` のいずれかであり、それぞれ gradient を（1）他の全 worker、（2）model-parallel group と直交する data-parallel group の worker と同期すべきこと、または（3）どの worker とも同期しないことを示す。例えば、model parallel の設定にかかわらず、gate network はすべての worker に複製される。attention layer は model-parallel sub-layer に分割される場合があるため、その tag は `data parallel` となる。各 worker は複数の固有 expert network を担当し、その tag は `none` である。*FastMoE* は、PyTorch 本来の distributed data parallel module に代えて、tag を識別して正しい同期を実行できるカスタム data parallel module を提供する。

<span id="section-4"></span>

## 4 高性能を達成するための最適化

単一ノード上の MoE 計算性能は重要であり、システムを任意の規模まで拡張する際の理論上限を決める。

MoE 層を計算する最も直感的な方法は、input batch を sample に分割し、一つずつ計算することである。その後、output feature を元の順序で stack する。しかし、単純な PyTorch operator で MoE モデルを実装しても高性能を得ることは難しいと分かった。達成できる性能は GPU の peak performance の $5\%$ 未満である。

<span id="figure-03"></span>

![異なる問題サイズでの GeMM 性能](../../papers/fastmoe/figure-03.png)

**図 3。** NVIDIA V100 上で `cuBLAS` を用いた、異なる問題サイズの `GeMM` 性能。

一般性を失うことなく、expert network を FFN と仮定する。FFN 内の主要な operator は fully-connected layer に由来し、複数の `GeMM` operator で構成されることに注意されたい。batch を単一 sample に分割すると、`GeMM` operation は `GeMV` へと退化する。[図 3](#figure-03) は、異なる batch size で sample fully-connected layer を実行したときの浮動小数点計算 throughput を示す。現代の heterogeneous compute device では、matrix multiplication operator が全次元に適用される高度な tiling technique によって細かく調整されている。そのため、batch size が十分に大きい場合に限って throughput が theoretical peak に近づくことは驚くに当たらない。このことから、MoE 計算で高性能を得るには、hardware resource を十分に利用できるよう sample を batch 化すべきだという原則が導かれる。

*FastMoE* は、同じ expert に向かうすべての input sample をまとめて batch 化する。data representation の制約により、*FastMoE* は専用開発した CUDA kernel で memory movement を行い、overhead を削減する。各 sample の行き先を示す gate index が与えられたとき、同じ gate に向かうすべての input sample を連続した memory space に配置する処理を `scatter` と呼ぶ。しかし、neural network の他の部分、例えば Transformer の attention layer では、batch を元の順序に編成する必要がある。expert が別の連続 memory space に出力した後で逆 operation を実行し、gate index に従って散布済みの feature vector を元の順序へ戻す。*FastMoE* ではこの処理を `gather` と呼ぶ。

<span id="figure-04"></span>

![MoE 層の並べ替え計算例](../../papers/fastmoe/figure-04.png)

**図 4。** MoE 層の並べ替え計算例

[図 4](#figure-04) に並べ替え計算の処理を示す。input sample から expert への割り当てが十分に均衡している場合、[図 3](#figure-03) に従えば、各 expert は十分な hardware utilization を実現できる比較的大きな input batch size を持つと期待される。しかし、input training data をランダムにサンプリングする性質上、load imbalance は必ず発生する。数百万回の training iteration の間に、一つの expert が受け取る input sample がごく少数となる可能性は高い。さらに、一つの worker に複数の expert を配置すると、expert の local batch size は統計的に見て、平均では data parallel の場合より小さくなる。*FastMoE* は custom stream manager を用いて複数 expert の計算を同時実行し、潜在的な throughput 向上を引き出す。

<span id="section-5"></span>

## 5 評価

本節では、単一 GPU 上で *FastMoE* の訓練速度を別の PyTorch MoE 実装 [Rau19] と比較する。また、分散訓練時の *FastMoE* のスケーラビリティを報告する。我々の知る限り、*FastMoE* は異なるノードと GPU にまたがって動作できる唯一の PyTorch ベース MoE システムである。さらに、*FastMoE* を用いて訓練した MoE Transformer モデルの end-to-end 性能も示す。

<span id="section-5-1"></span>

### 5.1 実験設定

計算 task を特徴付けるため、以下の記法を用いる。各 GPU に $n_e$ 個の expert を配置する。各 expert は、それぞれ $d_m\times d_h$ および $d_h\times d_m$ のサイズを持つ二つの linear layer を適用する。input は $n_b$ 個の sample を含む。gate module は、各 sample を各 expert が処理する適合度を評価する。各 input sample について、score が最も高い上位 $k$ 個の expert を選択して sample を処理する。

加えて、同じ計算を行うが結果には数えない warm-up round を複数回実施する。各実験では task を $16$ 回実行し、その平均時間を性能の計算に用いる。実行時間の標準偏差も調べた。いずれも無視できる大きさである。

<span id="section-5-2"></span>

### 5.2 単一 GPU 上の訓練速度

NVIDIA TESLA V100 PCIe GPU 上で、ベースライン [Rau19] の `MoE` module と同様の task を完了する `FMoETransformerMLP` の性能を測定する。ベースラインは純粋な PyTorch API で実装され、model structure は hard-code されている。比較の公平性を保つため、両 module とも一つの fully-connected layer からなる gate network の weight としてランダム初期化した matrix を使用する。expert も同じ計算を行う。

<span id="figure-05"></span>

![FastMoE とベースラインの計算時間比較](../../papers/fastmoe/figure-05.png)

*latency は $n_b=4096,d_m=1024,d_h=4096,k=2$ で測定した。*

**図 5。** *FastMoE* とベースライン実装の計算時間比較。

[図 5](#figure-05) に示すように、ベースライン実装は一貫して *FastMoE* より低速である。expert 数が増えるにつれ、ベースラインが forward 計算に費やす時間は大幅に増加する一方、*FastMoE* の latency は、[第 4 節](#section-4) で述べた custom stream manager により安定している。*FastMoE* は訓練を対象としているため、backward time は forward time の上に積み上げている。各 iteration に要する総時間でも、*FastMoE* がベースラインを上回ることを確認した。

<span id="section-5-3"></span>

### 5.3 Cross-GPU および Cross-node スケーラビリティ

ノードをまたぐ複数 GPU へ *FastMoE* を拡張した場合の性能を調べるため、$8$ ノードのクラスタで実験を行う。各ノードには NVIDIA Tesla V100 GPU を $1$ 基搭載する。クラスタは Infiniband EDR switch と $8$ 枚の HCA card で相互接続される。matrix multiplication operation の FLOPs を計算し、訓練 throughput を表す。

<span id="figure-06"></span>

![複数ノードの複数 GPU にまたがる FastMoE のスケーラビリティ](../../papers/fastmoe/figure-06.png)

*throughput は $n_e=4,n_b=4096,d_m=1024,d_h=4096,k=2$ で測定した。*

**図 6。** 複数ノードの複数 GPU にまたがる *FastMoE* のスケーラビリティ

[図 6](#figure-06) の結果によれば、*FastMoE* はノード間のスケーラビリティを示す。GPU 数を $2$ から $8$ へ増やすと、全体 throughput は $10$ `TFLOPs` から $25$ `TFLOPs` へ増加し、準線形に拡大する。$2$ GPU へ拡張すると性能は単一 GPU の半分になることが分かり、*FastMoE* が通信に制約されていることを示唆する。計算に使う GPU が増えると、より多くの expert が導入され、input sample を交換する粒度が小さくなるため、network 上の data transfer 効率が低下する。

結論として、*FastMoE* のスケーラビリティにより、複数ノードの複数 GPU を用いて大規模 MoE モデルを性能向上とともに訓練できる。ただし、throughput にはさらなる最適化の余地が残る。

<span id="section-5-4"></span>

### 5.4 FastMoE による End-to-end 性能向上

Megatron-LM [Sho19] を用いて $8$ GPU 上で 12-layer GPT モデルを訓練し、*FastMoE* による end-to-end 性能向上を検証する。[第 3 節](#section-3) で述べたように、MoE 構造には *FastMoE* の Megatron adapter を使用する。各層では $96$ expert を GPU 間に分散し、各 GPU に $12$ expert を配置する。各 input token について、score が最も高い上位 $2$ expert が処理に用いられる。expert MLP layer の $d_h$ を半分にし、gate が導入する無視できる追加 FLOPs を除いて、モデルの有効 FLOPs がほぼ同じになるようにする。baseline model と MoE model は、どちらも $70$ 時間訓練する。訓練中の `lm loss` metric は、モデルの収束傾向を示す。

<span id="figure-07"></span>

![FastMoE で GPT モデルを訓練したときの loss curve](../../papers/fastmoe/figure-07.png)

*細い濃色の線は元の loss curve を $0.97$ で指数平滑化したものであり、それぞれ明るく太い curve が元の curve を表す。*

**図 7。** *FastMoE* による GPT モデル訓練の loss curve

[図 7](#figure-07) から、ベースラインモデルの訓練速度は *FastMoE* の約 $3\times$ であることが分かった。*FastMoE* はより多くの計算と通信を行うため、これは妥当な速度低下である。幸い、同じ training iteration 数では MoE モデルの loss が大幅に低くなる。また、*FastMoE* の効率性により、同じ訓練時間でも MoE モデルはより低い loss を達成する。

<span id="section-6"></span>

## 6 まとめと今後の課題

本論文では、Mixture-of-Experts モデルを訓練するオープンソースシステム *FastMoE* を提示した。このシステムは一般的な PyTorch framework を基盤とし、現在は GPU 上の効率的な訓練をサポートする。異なるユーザーが MoE アーキテクチャのさまざまな側面を探索できるよう、複数レベルの使いやすいインターフェースを提供する。単一 GPU 上の *FastMoE* の性能は、GPU の能力を引き出せるよう十分に最適化されている。*FastMoE* は複数ノードの GPU にまたがって妥当なスケーラビリティで動作し、モデル規模をさらに拡大できる。*FastMoE* を用いた end-to-end モデル訓練実験では、実際のモデル性能上の優位性が確認された。

我々は、機能の追加と訓練の高速化に向けて、引き続き *FastMoE* を開発している。GShard モデル [Lep20] と比較すると、*FastMoE* には expert 間の load balancing をサポートする機能がない。load-balance monitor と load-balance loss のサポートを開発中である。また、MoE モデルの load や save など、utility 面でシステムをより使いやすくすることにも取り組んでいる。複数 GPU にまたがる性能には、高性能計算と機械学習の双方からの共同作業が必要である。オープンソースプロジェクトへのあらゆる貢献を歓迎する。参加を期待している。
