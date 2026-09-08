---
title: 'The Era of 1-bit LLMs'
createTime: 2026/09/08 13:02:00
permalink: /ja/papers/bitnet-b1-58/
pageClass: paper-reading
---

> [Shuming Ma](https://shumingma.com/) [+author-note]、[Hongyu Wang](https://ustcwhy.github.io/) [+author-note]、[Lingxiao Ma](https://xysmlx.github.io/)、[Lei Wang](https://dblp.org/pid/181/2817-222)、[Wenhui Wang](https://www.microsoft.com/en-us/research/people/wenwan/)、[Shaohan Huang](https://buaahsh.github.io/)、[Li Dong](https://dong.li/)、[Ruiping Wang](https://www.jdl.link/user/rpwang/index.htm)、[Jilong Xue](https://dblp.org/pid/06/10336)、[Furu Wei](https://www.microsoft.com/en-us/research/people/fuwei/) [+author-note]。2024 年 2 月 27 日に arXiv へ初回投稿。現行版は v1。進行中の研究である。[The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764)。<a href="/paper/bitnet-b1-58.pdf" target="_blank" rel="noopener noreferrer">原論文 PDF</a>。[DOI](https://doi.org/10.48550/arXiv.2402.17764)。[TeX ソース](https://export.arxiv.org/e-print/2402.17764v1)。正確な印刷レイアウトと参考文献については、原論文 PDF を正とする。

[+author-note]: Shuming Ma と Hongyu Wang は同等に貢献した。Furu Wei は責任著者である。Shuming Ma、Lingxiao Ma、Lei Wang、Wenhui Wang、Shaohan Huang、Li Dong、Jilong Xue、Furu Wei は Microsoft Research に所属する。Hongyu Wang と Ruiping Wang は中国科学院大学に所属する。[GeneralAI](https://aka.ms/GeneralAI)。

## 概要

BitNet [Wan23] などの最近の研究は、1-bit 大規模言語モデル（LLM）の新時代への道を開きつつある。本研究では、LLM のすべてのパラメータ（すなわち重み）が三値 $\{-1, 0, 1\}$ を取る、**BitNet b1.58** という 1-bit LLM の変種を導入する。モデルサイズと学習トークン数が同じ全精度（すなわち FP16 または BF16）Transformer LLM に対して、perplexity と最終タスク性能の両方で同等となる一方、レイテンシ、メモリ、スループット、エネルギー消費の面ではコストを大幅に削減する。より本質的には、1.58-bit LLM は、高性能かつ低コストな次世代 LLM を学習するための新しいスケーリング則とレシピを定義する。さらに、新しい計算パラダイムを可能にし、1-bit LLM 向けに最適化された専用ハードウェアを設計する道を開く。

<span id="figure-01"></span>

![BitNet b1.58 と全精度 Transformer LLM の Pareto 比較および計算パラダイム](../../papers/bitnet-b1-58/figure-01.png)

**図 1.** 1-bit LLM（BitNet b1.58 など）は、モデル性能を維持しながら LLM の推論コスト（レイテンシ、スループット、エネルギー）を削減する Pareto 解を与える。BitNet b1.58 の新しい計算パラダイムは、1-bit LLM 向けに最適化された新しいハードウェアの設計を求めるものである。

<span id="section-1"></span>

## 1 1-bit LLM の時代

近年、AI 分野では大規模言語モデル（LLM）のサイズと能力が急速に成長している。これらのモデルは幅広い自然言語処理タスクで顕著な性能を示してきたが、サイズの増大は展開上の課題を生み、高いエネルギー消費による環境面と経済面への影響も懸念されている。

これらの課題に対処する一つの方法は、学習後量子化を用いて推論用の低ビットモデルを作ることである [Xia23, Fra23, Che24b, Tse24]。この技術は重みと活性化の精度を下げ、LLM のメモリ要件と計算要件を大幅に削減する。16 bit から、4-bit 変種 [Fra23, Lin24] などのより低いビット数へ移行する傾向が続いている。しかし、学習後量子化は産業界の LLM で広く使われているものの、最適ではない。

BitNet [Wan23] など、1-bit モデルアーキテクチャに関する最近の研究は、性能を維持しながら LLM のコストを削減する有望な方向を示している。通常の LLM は 16-bit 浮動小数点値（すなわち FP16 または BF16）を用い、どの LLM でも処理の大部分は行列乗算である。したがって、計算コストの大半は浮動小数点の加算と乗算から生じる。対照的に、BitNet の行列乗算には整数加算しか含まれず、LLM のエネルギーコストを数桁削減できる。多くのチップでは電力が計算性能の根本的な上限となるため、エネルギーの節約は計算の高速化にもつながる。

計算に加えて、推論時にモデルパラメータを DRAM からオンチップアクセラレータのメモリ（SRAM など）へ転送する処理にも大きなコストがかかることがある。スループット向上のために SRAM を拡大する試みもあるが、DRAM より大幅に高いコストを招く。全精度モデルと比べると、1-bit LLM は容量と帯域幅の両面でメモリフットプリントがはるかに小さい。これにより、DRAM から重みを読み込むコストと時間を大幅に減らし、より高速で効率的な推論が可能になる。

本研究では、すべてのパラメータが三値 $\{-1, 0, 1\}$ を取る、**BitNet b1.58** という重要な 1-bit LLM の変種を導入する。元の 1-bit BitNet に値 0 を追加した結果、二進法で 1.58 bit となる。BitNet b1.58 は元の 1-bit BitNet の利点をすべて保持しており、その中には行列乗算に乗算処理をほとんど必要とせず、高度に最適化できる新しい計算パラダイムも含まれる。さらに、エネルギー消費は元の 1-bit BitNet と同等であり、FP16 LLM ベースラインと比べてメモリ消費、スループット、レイテンシの面ではるかに効率的である。BitNet b1.58 には、さらに二つの利点がある。第一に、モデルの重みに 0 を含めることで特徴フィルタリングを明示的にサポートできるため、モデリング能力が高く、1-bit LLM の性能を大幅に改善できる。第二に、同じ構成（モデルサイズ、学習トークン数など）を用いた場合、BitNet b1.58 は 3B サイズから、perplexity と最終タスク性能の両方で全精度（すなわち FP16）ベースラインと同等になることを実験で示す。

<span id="section-2"></span>

## 2 BitNet b1.58

BitNet b1.58 は BitNet アーキテクチャに基づいている。これは *nn.Linear* を *BitLinear* で置き換えた Transformer である。1.58-bit の重みと 8-bit の活性化を用いてゼロから学習する。元の BitNet と比べていくつかの変更を導入しており、以下に要約する。

**量子化関数。** 重みを -1、0、+1 に制約するため、*absmean* 量子化関数を採用する。まず重み行列を絶対値の平均でスケーリングし、次に各値を $\{-1, 0, +1\}$ のうち最も近い整数へ丸める。

<span id="equation-01"></span>

$$
\widetilde{W}=\mathrm{RoundClip}\left(\frac{W}{\gamma+\epsilon},-1,1\right),
$$

<span id="equation-02"></span>

$$
\mathrm{RoundClip}(x,a,b)=\max(a,\min(b,\mathrm{round}(x))),
$$

<span id="equation-03"></span>

$$
\gamma=\frac{1}{nm}\sum_{ij}|W_{ij}|.
$$

活性化の量子化関数は BitNet と同じ実装に従うが、非線形関数の前で活性化を $[0,Q_b]$ の範囲にスケーリングしない点が異なる。その代わりに、ゼロ点量子化をなくすため、トークンごとにすべての活性化を $[-Q_b,Q_b]$ へスケーリングする。この方法は実装とシステムレベルの最適化の両方で扱いやすく簡単であり、実験では性能への影響を無視できる程度に抑えられた。

**LLaMA 類似コンポーネント。** LLaMA [Tou23, Tou23a] のアーキテクチャは、オープンソース LLM の事実上の基盤となっている。オープンソースコミュニティに適合させるため、BitNet b1.58 の設計には LLaMA に似たコンポーネントを採用する。具体的には RMSNorm [Zha19]、SwiGLU [Sha20]、rotary embedding [Su24] を使用し、すべての bias を除去する。これにより、BitNet b1.58 は一般的なオープンソースソフトウェア（Huggingface、vLLM [Kwo23]、llama.cpp [+llama-cpp] など）へ最小限の作業で統合できる。

[+llama-cpp]: [llama.cpp](https://github.com/ggerganov/llama.cpp)。

<span id="table-01"></span>

![BitNet b1.58 と LLaMA LLM の perplexity、メモリ、レイテンシ](../../papers/bitnet-b1-58/table-01.png)

**表 1.** BitNet b1.58 と LLaMA LLM の perplexity およびコスト。

<span id="table-02"></span>

![BitNet b1.58 と LLaMA LLM の七つの最終タスクにおけるゼロショット精度](../../papers/bitnet-b1-58/table-02.png)

**表 2.** 最終タスクにおける BitNet b1.58 と LLaMA LLM のゼロショット精度。

<span id="section-3"></span>

## 3 結果

さまざまなサイズの BitNet b1.58 を、再現した FP16 LLaMA LLM と比較した。公平に比較するため、モデルを RedPajama データセット [Tog23a] の 1000 億トークンで事前学習した。ARC-Easy [Yad19]、ARC-Challenge [Yad19]、Hellaswag [Zel19]、Winogrande [Sak19]、PIQA [Bis20]、OpenbookQA [Mih18b]、BoolQ [Cla19] を含む一連の言語タスクでゼロショット性能を評価した。WikiText2 [Mer16] と C4 [Raf19] の各データセットにおける検証 perplexity も報告した。

LLaMA LLM と BitNet b1.58 の実行時 GPU メモリおよびレイテンシを比較した。測定には、GPU デバイス上の LLM 推論レイテンシ向けに十分最適化された FasterTransformer [+fastertransformer] コードベースを使用した。BitNet b1.58 には Ladder [Wan24e] の 2-bit カーネルも統合した。推論コストの主要部分であるため、出力トークン当たりの時間を報告した。

[+fastertransformer]: [NVIDIA FasterTransformer](https://github.com/NVIDIA/FasterTransformer)。

[表 1](#table-01) は BitNet b1.58 と LLaMA LLM の perplexity およびコストをまとめている。BitNet b1.58 はモデルサイズが 3B になると、perplexity の点で全精度 LLaMA LLM と同等になり始める一方、2.71 倍高速で、GPU メモリ使用量は 3.55 分の 1 である。特に、3.9B サイズの BitNet b1.58 は 2.4 倍高速で、メモリ消費を 3.32 分の 1 に抑えながら、3B の LLaMA LLM より大幅に高い性能を示す。

[表 2](#table-02) は最終タスクにおけるゼロショット精度の詳細な結果を示す。評価には *lm-evaluation-harness* [+lm-evaluation-harness] のパイプラインを用いた。結果から、モデルサイズが増えるにつれて BitNet b1.58 と LLaMA LLM の性能差が縮まることが分かる。さらに重要なことに、BitNet b1.58 は 3B サイズから全精度ベースラインの性能に並ぶ。perplexity での観察と同様に、最終タスクの結果でも、3.9B の BitNet b1.58 はメモリとレイテンシのコストを抑えつつ、3B の LLaMA LLM を上回る。この結果は、BitNet b1.58 が最先端の LLM モデルに対する Pareto 改善であることを示している。

[+lm-evaluation-harness]: [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)。

**メモリとレイテンシ** モデルサイズをさらに 7B、13B、70B へ拡大し、コストを評価した。[図 2](#figure-02) はレイテンシとメモリの傾向を示しており、モデルサイズの拡大に伴って高速化率が上昇する。特に、70B の BitNet b1.58 は LLaMA LLM ベースラインより 4.1 倍高速である。これは、*nn.Linear* の時間コストがモデルサイズとともに増えるためである。embedding は全精度のままであり、大きなモデルほどそのメモリ比率が小さくなるため、メモリ消費も同様の傾向を示す。レイテンシとメモリはいずれも 2-bit カーネルで測定しており、さらにコストを削減する最適化の余地が残っている。

<span id="figure-02"></span>

![モデルサイズを変えた BitNet b1.58 のデコードレイテンシとメモリ消費](../../papers/bitnet-b1-58/figure-02.png)

**図 2.** モデルサイズを変えたときの BitNet b1.58 のデコードレイテンシ（左）とメモリ消費（右）。

**エネルギー** BitNet b1.58 と LLaMA LLM の算術演算によるエネルギー消費も推定する。行列乗算が LLM のコストの大部分を占めるため、主にその計算に注目する。[図 3](#figure-03) はエネルギーコストの構成を示す。BitNet b1.58 の大部分は INT8 加算であるのに対し、LLaMA LLM は FP16 加算と FP16 乗算の両方からなる。[Hor14, Zha22g] のエネルギーモデルによれば、BitNet b1.58 は 7nm チップ上の行列乗算における算術演算のエネルギー消費を 71.4 分の 1 に削減する。さらに、512 トークンを処理するモデルのエンドツーエンドのエネルギーコストを報告した。モデルサイズが拡大するにつれ、BitNet b1.58 は FP16 LLaMA LLM ベースラインよりもエネルギー消費の点でさらに効率的になる。これは、モデルサイズとともに *nn.Linear* の割合が増え、大きなモデルでは他のコンポーネントのコストが小さくなるためである。

<span id="figure-03"></span>

![BitNet b1.58 と LLaMA LLM の算術演算およびエンドツーエンドのエネルギー消費](../../papers/bitnet-b1-58/figure-03.png)

**図 3.** 7nm プロセスノードにおける BitNet b1.58 と LLaMA LLM のエネルギー消費。左は算術演算のエネルギー構成、右はモデルサイズごとのエンドツーエンドのエネルギーコストを示す。

**スループット** 2 枚の 80GB A100 カード上で、70B パラメータの BitNet b1.58 と LLaMA LLM のスループットを比較する。70B の LLaMA LLM をこれらのデバイス上で実行できるよう、パイプライン並列化 [Hua19c] を使用した。系列長を 512 とし、GPU メモリの上限に達するまでバッチサイズを増やした。[表 3](#table-03) に示すように、70B の BitNet b1.58 は LLaMA LLM の最大 11 倍のバッチサイズに対応でき、スループットは 8.9 倍高くなる。

<span id="table-03"></span>

![70B パラメータの BitNet b1.58 と LLaMA LLM の最大バッチサイズおよびスループット](../../papers/bitnet-b1-58/table-03.png)

**表 3.** 70B の BitNet b1.58 と 70B の LLaMA LLM のスループット比較。

**BitNet b1.58 は、モデル性能と推論コストに関する新しいスケーリング則を可能にする**。 [図 2](#figure-02) と [図 3](#figure-03) の結果に基づけば、1.58-bit と 16-bit の異なるモデルサイズの間には、参考として次の対応関係が成り立つ。

- 13B BitNet b1.58 は、レイテンシ、メモリ使用量、エネルギー消費の点で 3B FP16 LLM より効率的である。
- 30B BitNet b1.58 は、レイテンシ、メモリ使用量、エネルギー消費の点で 7B FP16 LLM より効率的である。
- 70B BitNet b1.58 は、レイテンシ、メモリ使用量、エネルギー消費の点で 13B FP16 LLM より効率的である。

**2T トークンでの学習** 学習トークン数は LLM にとって重要な要因である。トークン数に関する BitNet b1.58 のスケーラビリティを検証するため、最先端のオープンソース 3B モデルである StableLM-3B [Tow23] のデータレシピに従い、2T トークンで BitNet b1.58 モデルを学習した。両モデルを、Winogrande [Sak19]、PIQA [Bis20]、SciQ [Wel17]、LAMBADA [Pap16]、ARC-easy [Yad19] からなるベンチマークで評価した。[表 4](#table-04) にゼロショット精度を示す。accuracy と normalized accuracy の両方で測定されるタスクでは、二つの平均を取る。2T トークンで学習した StableLM 3b の結果は、その技術報告から直接取得した。BitNet b1.58 はすべての最終タスクでより高い性能を達成しており、1.58-bit LLM にも強い汎化能力があることが分かる。

<span id="table-04"></span>

![2T トークンで学習した BitNet b1.58 と StableLM-3B のゼロショット結果](../../papers/bitnet-b1-58/table-04.png)

**表 4.** 2T トークンで学習した BitNet b1.58 と StableLM-3B の比較。

<span id="section-4"></span>

## 4 議論と今後の課題

**1-bit Mixture-of-Experts（MoE）LLM** Mixture-of-Experts（MoE）は、LLM に対してコスト効率のよい手法であることが示されている。計算 FLOPs を大幅に削減する一方、高いメモリ消費とチップ間通信のオーバーヘッドが展開と応用を制限する。これらの課題は 1.58-bit LLM によって解決できる。第一に、メモリフットプリントの削減により、MoE モデルの展開に必要なデバイス数が減る。さらに、ネットワーク経由で活性化を転送するオーバーヘッドも大幅に削減される。最終的に、モデル全体を単一チップへ配置できれば、オーバーヘッドはなくなる。

**LLM における長い系列のネイティブサポート** LLM の時代には、長い系列を扱う能力が重要な要件となっている。長い系列の推論における主な課題の一つは、KV キャッシュによるメモリ消費である。BitNet b1.58 は活性化を 16 bit から 8 bit へ削減し、同じリソースでコンテキスト長を 2 倍にできるため、長い系列のネイティブサポートへ向けた大きな一歩となる。1.58-bit LLM ではさらに 4 bit 以下へ可逆圧縮できる可能性があり、これは今後の課題とする。

**エッジおよびモバイル上の LLM** 1.58-bit LLM を用いることで、エッジおよびモバイルデバイス上の言語モデル性能を大幅に向上できる可能性がある。これらのデバイスはメモリと計算能力に制約があることが多く、LLM の性能と規模が制限される。しかし、1.58-bit LLM はメモリとエネルギー消費を削減するため、これらのデバイスへ展開でき、従来は実現できなかった幅広いアプリケーションが可能になる。これにより、エッジおよびモバイルデバイスの能力を大幅に高め、新しい LLM アプリケーションを実現できる。さらに、1.58-bit LLM は、エッジおよびモバイルデバイスの主要プロセッサである CPU にも適している。したがって、BitNet b1.58 をこれらのデバイス上で効率的に実行し、性能と能力をさらに向上できる。

**1-bit LLM のための新しいハードウェア** Groq [+groq] のような最近の研究は、LLM 専用ハードウェア（LPU など）の構築について有望な結果と大きな可能性を示している。さらに一歩進めて、BitNet [Wan23] が可能にした新しい計算パラダイムを踏まえ、1-bit LLM 向けに特化して最適化された新しいハードウェアとシステムの設計を構想し、行動を呼びかける。

[+groq]: [Groq](https://groq.com/)。
