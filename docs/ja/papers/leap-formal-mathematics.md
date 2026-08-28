---
title: 'LEAP for Formal Mathematics'
createTime: 2026/08/28 11:46:19
permalink: /ja/papers/leap-formal-mathematics/
pageClass: paper-reading
---

> [Po-Nien Kung](https://billkunghappy.github.io/ponien-kung/)、[Linfeng Song](https://scholar.google.com/citations?user=yWZdmLYAAAAJ)、[Dawsen Hwang](https://scholar.google.com/citations?user=yuX2FDAAAAAJ)、[Jinsung Yoon](https://scholar.google.com/citations?user=kiFd6A8AAAAJ)、[Chun-Liang Li](https://scholar.google.com/citations?user=vqHIt_sAAAAJ)、[Simone Severini](https://scholar.google.com/citations?user=yi-Q7zcAAAAJ)、[Mirek Olšák](https://dblp.org/pid/192/1864)、[Edward Lockhart](https://scholar.google.com/citations?user=P1MWvREAAAAJ)、[Quoc V Le](https://scholar.google.com/citations?user=vfT6-XIAAAAJ)、[Burak Gokturk](https://scholar.google.com/citations?user=351ivuQAAAAJ)、[Thang Luong](https://scholar.google.com/citations?user=Bmbkv6sAAAAJ)、[Tomas Pfister](https://scholar.google.com/citations?user=ahSpJOAAAAAJ)、[Nanyun Peng](https://violetpeng.github.io/)。2026 年 6 月 2 日に arXiv へ初回投稿。現行版は v2 で、2026 年 6 月 3 日に改訂された。[LEAP: Supercharging LLMs for Formal Mathematics with Agentic Frameworks](https://arxiv.org/abs/2606.03303v2)。[原論文 PDF](/paper/leap-formal-mathematics.pdf)。[DOI](https://doi.org/10.48550/arXiv.2606.03303)。[TeX ソース](https://export.arxiv.org/e-print/2606.03303v2)。正確な印刷レイアウトと参考文献については原論文 PDF を正本とする。

## 概要

大規模言語モデル（LLM）は非形式的な数学推論では高い能力を示すが、Lean のような形式言語で機械的に検証可能な証明を生成することは難しい。
本研究では、汎用基盤モデルに自動形式定理証明で最先端の性能を実現させるエージェント型フレームワーク、LEAP（LLM-in-Lean Environment Agentic Prover）を提案する。
LEAP は、非形式推論、指示追従、反復的な自己改善といった基盤モデルの能力を利用する。複雑な問題を小さな単位に分解し、Lean コンパイラと継続的にやり取りすることで、形式証明の構築と非形式ブループリントを結び付ける。
飽和が進む既存ベンチマークを超えて厳密に評価するため、Lean で形式化した IMO 形式の問題から成る Lean-IMO-Bench を導入する。問題文は短いが、証明はきわめて非定型的で複数段階に及び、難易度も幅広い。
北米の学部生を対象とする年次数学競技会である最新の 2025 Putnam Competition において、LEAP は全 12 問を解き、最先端の形式数学モデルによる近年の成果に並んだ。Lean-IMO-Bench では、汎用 LLM のワンショット形式証明成功率を 10% 未満から 70% へ高め、IMO 金メダル級の専用システムが記録した 48% を大きく上回った。さらに、組合せ論の未解決課題に対する複雑な証明を自律的に形式化し、Knuth による偶数位数 Cayley グラフの Hamilton 分解における重要な部分問題の検証済み証明を含め、研究水準での LEAP の有用性を示す。

<span id="section-1"></span>

## 1 はじめに

大規模言語モデル（LLM）は、「非形式数学推論」とも呼ばれる自然言語による数学推論で目覚ましい進歩を遂げ、数学競技と研究水準の数学の双方で、複雑な推論と問題解決に高い能力を示してきた [Hua25h, Luo25i, Fen26, Fen26a, Fen26b]。しかし、Hilbert [Var25] や Goedel-Prover-V2 [Lin25j] などの近年の研究で論じられているように、自然言語の解答には論理的誤謬やハルシネーションがしばしば含まれ、自動検証も難しい。この検証の難しさは自動化システムだけの限界ではない。人間の数学者にとっても、複雑な証明の検証は、希少な専門家の労力を必要とすることで知られる非常に時間のかかる作業である [Gre24]。有名な例に Kepler 予想の証明 [Hal05] があり、4 年間の査読を経ても査読者は正しさを「99% 確信している」としか述べられず [+kepler-conjecture]、最終的には 10 年に及ぶ形式検証が必要になった [Hal17]。この検証上のボトルネックは、自然言語で正しさを評価すること自体が難しいと示しており、形式数学を探究する動機になる。形式数学では、証明を機械検査可能な言語で記述し、Lean [Mou21]、Isabelle [Nip02]、Coq [Hue97]、HOL Light [Har09] のような厳密なカーネルで検証するため、正確性が保証された自動検証が可能である。それでも形式定理証明との隔たりを埋めることは大きな課題であり、現在の自動形式証明器の性能は、自然言語を用いる汎用 LLM より大幅に低い。

[+kepler-conjecture]: [Kepler 予想](https://en.wikipedia.org/wiki/Kepler_conjecture)。

この隔たりを埋めるため、研究コミュニティの近年の取り組みでは、汎用 LLM は専用化なしには厳密な形式タスクに有効でないという前提のもと、AlphaProof [Hub26]、DeepSeek Prover V2 [Ren25b]、Seed Prover [Che25h]、Goedel Prover V2 [Lin25j] などの専用証明器モデルを形式コーパスで微調整することが中心となっている。実際、FormalProofBench [Rav26] [+formalproofbench] と TaoBench [Tay26] によれば、汎用 LLM は専用証明器モデルより低い性能にとどまる場合が多い。

[+formalproofbench]: この論文には、ライブリーダーボードを備えた非公開データセット [https://www.vals.ai/benchmarks/proof_bench](https://www.vals.ai/benchmarks/proof_bench) が関連付けられている。リーダーボードへの参加を求めて数回連絡したが、返答はなかった。

近年の一部の研究ではエージェント型探索や推論時探索を検討しているが、依然として専用モデルに依存している。たとえば Hilbert [Var25]、AlphaProofNexus [Tso26]、Aristotle [Ach25]、Seed Prover V1.5 [Che25a] は、非形式推論には汎用 LLM を使う一方、Lean の証明段階には専用モデルを用いる。Axiom [+axiom] と Numina [+numina] は Putnam 2025 で高い成績を主張しているが、非公開のままでアクセス手段もなく、科学的に検証できない。

[+axiom]: [https://github.com/AxiomMath/putnam2025](https://github.com/AxiomMath/putnam2025).

[+numina]: [https://github.com/project-numina/Numina-Putnam2025](https://github.com/project-numina/Numina-Putnam2025).

本論文では、汎用 LLM のワンショット定理証明にはなお限界があるものの、ボトルネックは言語理解ではなく、長く複雑で正しい証明を一度に生成することだと示す。汎用 LLM は、強い非形式推論、指示追従、ツール利用、自己改善という、専用モデルを補完する能力を持つ。これらの能力は、証明構築を分解して反復的に改善するエージェント型 ATP フレームワークに適している。
そこで、形式数学に*汎用* LLM だけを用いるエージェント型フレームワーク、**LEAP（LLM-in-Lean Environment Agentic Prover）**を提案する。LEAP は人間の作業手順にならい、有向非巡回グラフ（DAG）を構成する高水準ブループリントを生成した後、Lean 証明を作り、コンパイラのフィードバックを用いて誤りを反復的に修正する。

MiniF2F [Zhe22a] や PutnamBench [Tso24] のような飽和したベンチマークを超えて進歩を評価するため、難度の高い非形式数学ベンチマーク IMO-Bench [Luo25] の問題文を Lean で形式化した Lean-IMO-Bench を導入する。既存のベンチマークは短い問題に焦点を当てるか、学部数学を幅広く扱うのに対し、Lean-IMO-Bench はそれらを補完する領域を対象とする。すなわち、初等的な問題文でありながら、解答にはきわめて非定型的な洞察が必要で、長く多段階かつ構造的に入り組んだ証明となる問題であり、形式定理証明をより厳しく試す。

実験では、北米で毎年開催される難度の高い学部数学競技会であり、2025 年の最高点が 120 点中 110 点、中央値がわずか 2 点だった最新の 2025 Putnam Competition において、LEAP は Lean で全 12 問を解き、満点の成績を収めた。これは Axiom [+axiom] や Numina [+numina] など、最先端の形式数学推論モデルによる近年の画期的成果に並ぶ。
Lean-IMO-Bench では、LEAP は汎用 LLM の成功率を 10% 未満から 70% へ大幅に高め、専用 ATP モデル（5%）と、2025 年 IMO で金メダル相当の得点を達成した専用 ATP コンポーネントを持つ強力なシステム Aristotle（48%）を上回った。
本論文の貢献は次の 3 点である。

- **作業手順に着想を得たエージェント設計** 人間の数学的作業手順をコード化し、高水準のブループリント作成、低水準の形式証明生成、コンパイラのフィードバックによる反復を組み合わせたエージェント型フレームワーク LEAP を提案する。重要なのは、LEAP が*汎用* LLM だけで最先端の形式定理証明を達成できると示し、専用の微調整が不可欠だという見方に疑問を投げかけた点である。
- **Lean-IMO-Bench データセット：** MiniF2F や PutnamBench のような飽和したベンチマークを超えて進歩を評価するため、IMO-Bench の非形式的な問題文を形式的な Lean 文へ翻訳した、新しく難度の高い Lean-IMO-Bench を導入する。リソースは [https://imobench.github.io](https://imobench.github.io) で公開している。
- **強力な実験結果と知見：** LEAP は Putnam 2025 の全 12 問を解き、Lean-IMO-Bench でも従来のベースラインを大幅に上回った。分析から、汎用 LLM による形式数学の主要なボトルネックは形式言語の理解だけではなく、証明環境との構造化された反復的なやり取りが欠けていることだと考えられる。LEAP が生成した Lean の解答は [https://github.com/google-deepmind/superhuman/tree/main/leap](https://github.com/google-deepmind/superhuman/tree/main/leap) で公開している。

<span id="section-2"></span>

## 2 LEAP：ブループリント駆動の自動定理証明

<span id="section-2-1"></span>

### 2.1 ブループリントによる証明の形式化

数学証明の形式化が一度で済むことはほとんどなく、高水準の議論を Lean へ段階的に翻訳するための構造化された計画が必要である。この複雑さを扱うため、近年の形式化では Lean Blueprint ツール [+lean-blueprint] がよく用いられる。このツールを使うと、数学者は Lean コードと連携した人間可読の証明ロードマップを記述し、各ノードが証明義務を表す有向非巡回グラフ（DAG）として可視化できる。
この作業手順は、Fermat の最終定理の形式化ロードマップ [+flt-roadmap] のような大規模プロジェクトの調整で重要な役割を果たしており、数年にわたる証明作業を明示的な依存グラフで構成している。

[+lean-blueprint]: [Lean Blueprint](https://github.com/PatrickMassot/leanblueprint).

[+flt-roadmap]: [Lean による Fermat の最終定理の形式化](https://leanprover-community.github.io/blog/posts/FLT-announcement/)。

この作業手順に着想を得て、階層的な分解と計画によって自動定理証明を行うエージェント LEAP を提案する。LEAP は一度で完全な証明を合成するのではなく、ブループリントを段階的に作成し、Lean のゴールを補助補題へ分解し、発展する証明計画を AND-OR DAG として維持する。

<span id="figure-01"></span>

![図 1．LEAP の作業手順](../../papers/leap-formal-mathematics/figure-01.png)

**図 1．** **LEAP の作業手順。** LEAP はまず、コンパイラフィードバックによる修正と LeanSearch [Gao24g] の検索を組み合わせた直接形式化を試みる。失敗した場合は非形式ブループリントと形式証明スケッチを生成し、依存関係が非巡回のまま保たれる場合に限り、検証済みのサブゴールを DAG に追加する。

<span id="section-2-2"></span>

### 2.2 全体像

[図 1](#figure-01) に LEAP の作業手順を示す。入力定理を受け取ると、LEAP はその Lean 文を根の*ゴール* [+goal] として登録し、AND-OR DAG の OR ノードで表す。未解決ゴールを処理する際、*状態リーダー*がその文、依存関係、関連補題を取得する。次に LEAP は、非形式証明を生成して Lean コードへ翻訳し、候補を Lean コンパイラで検査する**直接証明**をまず試みる。

[+goal]: *ゴール*とは証明すべき任意の定理文または補題文であり、分解によって*サブゴール*が導入される。[第 10 節](#section-10)を参照。

直接証明に失敗すると、LEAP は**分解**へ移る。最初に中間補題を提案する非形式ブループリントを作り、次にブループリントを Lean の証明スケッチへ翻訳する。スケッチは提案された補題だけを仮定して現在のゴールを証明する。主定理の本体には `sorry` を含めず、`sorry` プレースホルダは新たに提案した補題文にだけ許される。
スケッチが Lean コンパイラに受理されると AND ノードとして追加され、提案された補題は子 OR ノードとして追加される。これにより、すべての子サブゴールが証明されれば親ゴールも証明される。検証済みスケッチは*状態ライター*へ渡され、更新が非巡回性を保つことを確認してから DAG にコミットされる。その後、エージェントは新しく作られたサブゴールを再帰的に処理する。

この作業手順は、密接に結び付いた 3 つの設計判断に基づく。**DAG ベースの階層的メモ化**は進捗を保持して分岐間で補題を再利用し、**非形式・形式を交互に用いる計画**は自然言語の戦略と実行可能な Lean コードを結び付け、**検証誘導型証明探索**はコンパイラのフィードバックと LLM によるレビューを使って候補分岐を受理、修正、分解、または破棄する。

<span id="section-2-3"></span>

### 2.3 DAG による階層的メモ化

LEAP は AND-OR DAG を証明の進捗記録だけでなく、階層的メモ化の構成にも用いる。OR ノードは未解決ゴールまたは補題文を表し、それぞれ任意の有効な証明戦略で解決できる。AND ノードは候補分解を表し、その成功には構成するすべてのサブゴールの証明が必要となる。[図 2](#figure-02) にこの構造を示す。

DAG には 2 つの主要な利点がある。第 1 は**単調な詳細化**である。ゴールを補助サブゴールへ分解した後は、確立済みの依存順序を組み替えることなく、後続の探索を子孫ノードの展開と解決に集中できる。これにより、局所的な証明探索と大域的な証明構成が分離される。個々の証明試行は修正、展開、破棄できる一方、DAG は証明計画全体の安定した依存構造を保持する。第 2 は**補題メモ化**である。中間補題文は共有証明ノードとして保存され、同じ部分問題が異なる分岐で現れたときに再利用できる。これは*先行的補題計画*も可能にする。ブループリント生成時に、LEAP は現在のスケッチにはすぐ必要でなくても、後の証明段階を支え得る補助補題文を提案できる。このような将来用の補題は、現在の AND ノードの解決には不要なままグラフメモリで利用可能に保たれる。これらの性質により、独立した証明計画を共通の依存先へ合流させ、冗長な導出を減らせる。

得られた依存構造は透明性も高め、未解決のゴール、解決済みの補題、下流の進捗を妨げるノードを明らかにする。これにより LEAP は、追加補題、修正した分解、より強い仮定が必要な箇所を特定しやすくなる。また、人間と AI の協働に向けた解釈可能なブループリント型ワークスペースも提供する。

<span id="section-2-4"></span>

### 2.4 非形式・形式を交互に用いる計画

[図 1](#figure-01) に示すように、LEAP の直接証明経路とブループリント分解経路はいずれも非形式証明スケッチを経由する。これは LLM と Lean の相補的な長所を反映している。LLM は非形式推論、戦略生成、改善に有効であり、Lean は厳密で機械検査可能な検証を提供する。

直接証明では、LEAP は現在のゴールに対する非形式的な議論をまず生成し、その後に候補 Lean 証明へ翻訳する。分解では、ゴールを補助サブゴールへ帰着する方法を説明した非形式ブループリントを作り、提案した依存関係を記録する Lean スケッチへ変換する。どちらの場合も、非形式スケッチが形式化前の計画空間となり、コードだけを直接生成する場合より証明構築が壊れにくくなる（非形式証明とブループリントの例は[第 10 節](#section-10)を参照）。

この交互利用により、証明の進捗も解釈しやすくなる。各形式化試行に非形式的な根拠が対応するため、ユーザーは Lean コードやコンパイラのフィードバックだけでなく、証明段階や分解が提案された理由も調べられる。

<span id="section-2-5"></span>

### 2.5 検証誘導型証明探索

[図 1](#figure-01) に示すように、LEAP は 2 つの水準で検証を用いる。第 1 に、Lean コンパイラが候補証明とスケッチを形式的に検査し、受理したコードが構文的に有効で型も正しいことを保証する。証明スケッチについては、提案したサブゴール（補題）にだけ `sorry` プレースホルダを許す。これにより証明 DAG の AND-OR 意味論が保たれ、参照するすべてのサブゴールが証明されれば親ゴールも証明される。
第 2 に、ブループリントが新しいサブゴールを提案すると、LLM レビュアーが分解の品質を評価する。すなわち、サブゴールが親ゴールに関係するか、問題を容易にするか、証明を完成させる妥当な経路を与えるかを判断する。この計画水準のレビューは、構文的に有効な Lean スケッチが不適切なサブゴールや元の文より簡単でないサブゴールを導入し得る複雑なゴールで重要になる。このフィルタがなければ、エージェントは弱いブループリントを繰り返し展開し、実質的な進捗のない分岐に探索予算を費やす可能性がある。[第 5.3 節](#section-5-3)では LLM レビュアーを除いたアブレーションによって、この失敗モードを調べる。

したがって、LLM レビュアーは探索フィルタとして機能し、見込みのない分解を特定してバックトラックを起動し、別の戦略の探索を促す。現在の LEAP は、バックトラックを伴う単純な DFS を DAG 上で用いる。このレビュアーの有効性は、より広い将来の方向性も示す。LLM は形式証明空間の探索を導くヒューリスティック評価器としても利用できる可能性がある。

<span id="figure-02"></span>

![図 2．Putnam 2025 問 A6 の DAG 例](../../papers/leap-formal-mathematics/figure-02.png)

**図 2．** **Putnam 2025 問 A6 の DAG 例。** LEAP は定理を証明スケッチと補助補題へ分解する。**先行的補題計画**により、エージェントはすぐには必要でなくても後で役立ち得る補助補題文も提案できる。これらは破線の辺で示され、主定理の証明には不要である。緑色のノードは証明済みノード、茶色のブロックはノードで導入された定義、構造、変数を表す。

<span id="section-3"></span>

## 3 Lean-IMO-Bench：Lean による IMO 問題の形式化

<span id="table-01"></span>

![表 1．Lean-IMO-Bench の 3 評価タスクにおけるベースライン性能](../../papers/leap-formal-mathematics/table-01.png)

**表 1．** Lean-IMO-Bench の 3 評価タスクにおけるベースライン性能。自然言語証明の性能は人間の専門家によるレビューに基づく。

<span id="section-3-1"></span>

### 3.1 Lean-IMO-Bench

[Luo25] の基礎研究を発展させ、精選した 60 問から成る Lean-IMO-Bench を導入する。[Luo25] は、数学者と IMO メダリストから成る専門家委員会が審査した厳密な問題群 IMO-ProofBench を導入した。
本ベンチマークは 60 問を *Basic* と *Advanced* に均等に分け、各集合に 30 問を収録する。*Basic* は IMO 前段階から IMO-Medium までの難度を扱い、代数 8 問、組合せ論 8 問、数論 8 問、幾何 6 問を含む。*Advanced* は IMO-Hard までの新規問題を含み、代数 8 問、組合せ論 8 問、数論 6 問、幾何 8 問で構成される。全体として、代数、組合せ論、幾何、数論の間でおおむね均衡している。

Lean-IMO-Bench の精度を最大限に高めるため、Lean の専門家が全 60 問の問題文を手作業で形式化し、検証した。問題は IMO 水準であり、必要な数学的背景は初等的である。そのため、対応する Lean の解答は簡潔になると予想され、複雑な現代数学理論を形式化する際の負担を意図的に除いている。

このデータセットでは、**自然言語証明**、**形式定理証明**、**形式証明翻訳**という 3 つの異なるタスクでモデルを評価できるが、本論文では形式定理証明に焦点を当てる。Lean-IMO-Bench におけるベースライン性能を[表 1](#table-01)にまとめる。
自然言語証明タスクについては [Luo25] を参照する。Gemini 2.5 Pro は非形式推論で高い性能を示す。しかし[表 1](#table-01)のとおり、この性能は形式定理証明へ直接移行しない。Gemini 3.1 Pro は形式定理証明、特に Advanced 集合で大幅に低い性能となる。形式証明翻訳タスクで正しい非形式証明を与えても改善は小さく、Pass@128 は変わらず、Average@128 がわずかに上がるだけである。

[表 1](#table-01)は、モデルの Lean 能力に大きな隔たりがあることを示す。モデルはすでに自然言語でこれらの問題を解けるため、数学推論はボトルネックではなく、有効な Lean コードを安定して生成することが主要な課題である。

<span id="section-4"></span>

## 4 実験結果

バックエンドの大規模言語モデルに Gemini-3.1-pro を用いて LEAP を評価し、4 つのベースラインと比較する。**Gemini-3.1-pro** は強力な汎用モデルによるワンショット証明生成を試す。**Goedel-Prover-V2-32B** [Lin25j] は Lean に特化した最先端のオープンソース ATP モデルである。**Hilbert** [Var25] は Goedel-Prover-V2-32B と Gemini-3.1-pro を組み合わせたエージェント型 Lean 形式化フレームワークである。**Aristotle** [Ach25] は専用 ATP コンポーネントを備え、2025 年 IMO で金メダル級の性能を達成した専用自動定理証明システムである。

形式証明能力を **Putnam 2025** と提案する **Lean-IMO-Bench** の 2 データセットで評価する。Putnam 2025 は、難度の高い北米の数学競技会である第 86 回 William Lowell Putnam Mathematical Competition [+putnam-results] の学部水準の問題 12 問を含む。2025 年大会の最高点は 120 点中 110 点、平均点は約 10 点、中央値は 2 点だった。

[+putnam-results]: Mathematical Association of America、[*第 86 回 William Lowell Putnam Mathematical Competition の結果*](https://maa.org/news/results-of-the-86th-william-lowell-putnam-mathematical-competition/)。

<span id="section-4-1"></span>

### 4.1 Putnam 2025 の結果

[表 2](#table-02)に Putnam 2025 ベンチマークの評価結果を示す。Pass@128 設定では、直接形式化のベースライン（Gemini-3.1-pro と Goedel-Prover-V2-32B）は 1 問も解けず、単一パスの生成ではこのデータセットの論理的複雑さに不十分であると分かる。

<span id="table-02"></span>

![表 2．Putnam 2025 の結果](../../papers/leap-formal-mathematics/table-02.png)

**表 2．** Putnam 2025 の結果。緑のチェック（✓）は問題を解けたことを、赤のバツ（×）は失敗を示す。評価設定では、$^\diamond$ は pass@128、$^\dagger$ は rollout=2 を表す。

オープンソースのエージェント型フレームワーク Hilbert は直接生成より改善し、12 問中 4 問を解いた。しかし評価では、Hilbert の再帰探索設計が $\mathcal{O}((n \cdot b)^{d})$ という指数時間計算量を生むことが分かった。ここで $n$ は補題の再試行回数、$b$ は平均分岐係数、$d=10$ は最大証明深さである。この方法は冗長な LLM 呼び出しを大量に必要とするため、Hilbert の各 rollout に 7 日間の制限時間を設定した。最先端の非公開システムとの比較のため、Aristotle の性能も報告する。このシステムは非公開だが強力なベースラインであり、2 回の rollout で 12 問中 9 問を解いた。 [+aristotle-report]

[+aristotle-report]: [非公式報告](https://www.reddit.com/r/mlscaling/comments/1pjnccr/aristotle_smashes_putnam_by_solving_formally/)では Aristotle がこのベンチマークの 12 問中 10 問を解いたとされる。ただし、その報告の実行でも本研究の評価でも、問 A5 は解けなかった。

LEAP は Putnam 2025 の全 12 問を解き、直接形式化では 0% だったベンチマーク成功率を、エージェント型フレームワークによって 100% へ高めた。この性能は、Hilbert のような標準的再帰フレームワークで見られた探索上のボトルネックを解消する、LEAP のブループリント着想型 AND-OR DAG アーキテクチャから直接得られる。階層的メモ化を支えることで、LEAP は独立した証明分岐に共有中間補題を再利用させ、指数的な探索複雑性を大幅に緩和して問題を効率よく解ける。これらの結果に必要だった計算コストと探索効率の問題別内訳については、[表 3](#table-03)の実行時間と効率の統計を参照されたい。

<span id="table-03"></span>

![表 3．Putnam 2025 における LEAP の実行時間と探索効率](../../papers/leap-formal-mathematics/table-03.png)

**表 3．** **Putnam 2025 における LEAP の実行時間と探索効率。** 各問題について、計算コスト（検証済み証明までの LLM 呼び出し総数）、探索した空間（有効な DAG ノード／補題）、最終的な Lean 証明の行数を報告する。

<span id="section-4-2"></span>

### 4.2 Lean-IMO-Bench の結果

[表 4](#table-04)に Lean-IMO-Bench の評価結果を示す。このデータセットは、より幅広い数学分野と異なる複雑度の階層にまたがるモデルの頑健性を試し、Putnam ベンチマークを補う課題として採用した。

直接形式化のベースライン（Gemini-3.1-Pro と Goedel-Prover-V2-32B）およびオープンソースの Hilbert フレームワークは、このデータセットで大きく苦戦し、Advanced 集合では性能が著しく低下する。非公開の Aristotle システムは Basic 問題の大半を解くものの、複雑さが増すと有効性が急激に下がる。特に、評価したすべての方法で Geometry 分野の性能はほぼ 0 のままである。これは、補助的な分野専用フレームワークなしに Lean でオリンピック水準の幾何を形式化することが難しいという既知の事実に合致する。この分野は、極端な形式化制約下で汎用推論を評価するためだけに残している。

これらのベースラインに対し、LEAP は全体で最も高い成功率を達成し、Basic 集合で 83.3%、Advanced 集合で 56.7% を記録した。DAG ベースのアーキテクチャを有効に活用することで、LEAP は高い分野汎化性能を示し、難度の階層に関係なく Algebra と Number Theory の両方で成功率 100% を維持した。

<span id="table-04"></span>

![表 4．Lean-IMO-Bench の結果](../../papers/leap-formal-mathematics/table-04.png)

**表 4．** Lean-IMO-Bench の結果。**Basic** と **Advanced** の各集合について、数学分野別の成功率（%）を報告する。評価設定では、$^\diamond$ は pass@128、$^\dagger$ は rollout=2 を表す。各部分の最良結果を太字で示す。

<span id="section-5"></span>

## 5 考察

<span id="section-5-1"></span>

### 5.1 ワンショット形式化を超えて

LEAP の中心的な動機は、汎用基盤モデルが Lean 専用証明器でなくても、有効な反復的形式化器になり得るという点にある。専用証明器は形式証明合成のために訓練される一方、汎用モデルは指示追従、長文脈推論、非形式計画、ツール利用、フィードバックに基づく修正といった相補的な能力を持つ。

この効果を切り分けるため、[図 1](#figure-01)に示した*直接形式化*コンポーネントを 2 つの設定で評価する。ワンショット設定では、独立に標本化した証明試行に対して各モデルを Pass@128 で評価する。反復設定では、各モデルに初回試行を 1 回だけ与え、その後にコンパイラフィードバックによる修正を最大 20 段階行い、より小さな標本化予算で Pass@1 の結果を得る。[表 5](#table-05)に示すように、Goedel-Prover-V2-32B はこのフィードバックループから恩恵を受けないが、Gemini-3.1-pro は $20.0\%$ から $36.6\%$ へ大幅に改善する。

これは、反復的形式化が局所的な Lean 証明合成以外の能力にも依存することを示唆する。コンパイラエラーの解釈、文脈の維持、複数段階にわたる証明試行の修正は、ワンショット形式証明の正確さと同じほど重要になり得る。この結果は、LEAP の推論中核に汎用基盤モデルを使うことを支持すると同時に、局所的な証明生成で専用証明器と組み合わせる可能性も残している。

<span id="table-05"></span>

![表 5．ワンショット形式化と反復的形式化の性能](../../papers/leap-formal-mathematics/table-05.png)

**表 5．** Lean-IMO-Bench Basic 集合における**ワンショット**形式化と**反復的形式化**の性能。

<span id="section-5-2"></span>

### 5.2 DAG ベースのメモ化の効果

LEAP は、孤立した分解木ではなく DAG ベースのメモリとして証明の進捗を維持する。これにより、中間補題を共有ノードとして保存して分岐間で再利用しながら、既存のゴール、依存関係、以前に提案した補題などのグラフ文脈を参照できる。

この設計を評価するため、同じ作業手順に従いながら大域的な補題共有を除いた木構造版と LEAP を比較する。[表 6](#table-06)に示すように、木構造版だけでも、Basic と Advanced でそれぞれ 36.6% と 6.6% を記録した Hilbert [Var25]（[表 4](#table-04)）を大幅に上回る。これは、DAG ベースのメモ化がなくても、非形式・形式を交互に用いる計画と検証誘導型探索が有効だと示している。完全な DAG 版は Basic で 73.3% から 83.3%、Advanced で 40.0% から 56.7% へさらに性能を高め、大域的証明メモリの利点を示す。

改善は Advanced Algebra や Advanced Number Theory のような難しい分野で特に顕著であり、そこでは共有補題とグラフ文脈が重要になりやすい。この向上は 2 つの効果によると考えられる。第 1 に、DAG は先行的補題計画を支え、高水準ノードで提案された補助補題を後から下流のサブゴールが再利用できる（[図 2](#figure-02)）。第 2 に、繰り返し現れる部分問題を分岐間で共有でき、同じ補題を何度も再発見または再証明せずに済む。これらの性質が冗長な導出を減らし、証明探索の効率を高める。

<span id="table-06"></span>

![表 6．DAG メモ化のアブレーション](../../papers/leap-formal-mathematics/table-06.png)

**表 6．** **DAG メモ化のアブレーション。** Lean-IMO-Bench Basic（B）／Advanced（A）集合における分野別成功率（%）。

<span id="section-5-3"></span>

### 5.3 LLM 誘導型証明探索に向けて

コンパイラ検証は証明スケッチが形式的に正しく型付けされているかを確認するが、その分解が有用かどうかは確認しない。スケッチは、役に立たない、難しすぎる、または元のゴールとほぼ同じ提案補題から親ゴールを証明することがある。LEAP では、LLM レビュアーが局所探索ヒューリスティックとして働き、候補分解を DAG にコミットする前に、親ゴールを意味のある形で簡単にするかを判断して選別する。

このアブレーションは Putnam 2025 問 A5 に焦点を当てる。これは評価で最も難しい事例の 1 つであり、LEAP が証明の形式化に成功するまで最長の実行時間と 2 回の rollout を要したからである。LLM ベースの分解レビュアーを除くと、エージェントは 8 回 rollout を試しても失敗した。この対比は、局所的な LLM レビューが有用な探索信号を与えることを示唆する。弱い分解を早期に拒否し、バックトラックを起動し、実質的に進展しない分岐へ rollout を費やすことを防ぐ。
さらに、アブレーション設定での分解トレースを調べ、代表的な失敗例を[図 3](#figure-03)に示す。
この分解は形式的には許容されるが、数学的状態を簡単にしない。エージェントはまず祖父ゴールの定義を展開して中間補題を作り、その後に定義を折り畳んで、元の文と構文的に同一のサブゴールを提案する。意味論的レビューがなければ、この重複補題は新しい段階として扱われ、探索予算を使い果たすまで同じ非生産的な分解を繰り返す。この失敗は LLM 誘導型証明探索の可能性を示す。レビュアーは、提案補題が実際に証明を進めるかを評価し、循環する分岐や単純化しない分岐を刈り込み、より有望な経路へ計算資源を向けられる。

<span id="figure-03"></span>

![図 3．LLM レビューがない場合の非生産的な分解](../../papers/leap-formal-mathematics/figure-03.png)

**図 3．** **LLM レビューがない場合の非生産的な分解。** 提案サブゴールは祖父ゴールを言い換えたものなので、分解は形式的には許容されるが、証明探索を簡単にしない。

<span id="section-5-4"></span>

### 5.4 展望：形式証明器としての汎用 LLM：ゼロから最先端へ

LEAP が示したように、汎用 LLM の低いワンショット定理証明性能と最先端の結果との間にある、一見埋めがたい隔たりは、適切に設計したエージェント型フレームワークによって効果的に埋められる。小型の専用 LLM だけに依存する枠組みから転換することで、基盤モデルが持つ広範な知識、指示追従、自己修正能力で十分以上だと示した。適切な足場を与えれば、基盤モデルは形式数学でほぼゼロの性能から、きわめて複雑な問題を解くまで進歩できる。

小型の専用 LLM には基盤モデルのような包括的なエージェント能力がないが、それでも価値があると考える。基盤モデルによる高水準の構造的推論と、微調整した専用モデルによる集中的な形式段階の生成を組み合わせたハイブリッドアーキテクチャは、非常に有効な設計パターンになり得る。ただし、本研究の主目的はエージェント型作業手順における汎用 LLM 単体の力を明らかにすることなので、このハイブリッド手法の検討は本論文の範囲外とする。

<span id="section-6"></span>

## 6 ケーススタディ：組合せ論の未解決問題の形式化

**有向 Cayley グラフの Hamilton 分解。** きわめて複雑な数学タスクで LEAP を評価するため、最近解決された組合せ論の未解決問題、すなわち偶数 $m$ に対する有向 Cayley グラフ $\Gamma_{m}=Cay(\mathbb{Z}_{m}^{3},\{e_{1},e_{2},e_{3}\})$ の Hamilton 分解を対象とした。Donald Knuth が最初に提示したこの問題は、グラフの有向弧を、互いに異なる全域 Hamilton サイクルちょうど 3 本へ分割できるかを問う。偶数の場合の構成に対する非形式数学証明はきわめて入り組んでおり、大量の組合せ解析と、グラフの異なる層にまたがる局所的な欠陥ルーティングに依存する。
形式化では重要な部分問題に焦点を当てた。単一の色クラスのルーティング力学を二次元平面へ射影すると、長さ $m^{2}$ の途切れない数学的サイクルになることを厳密に検証する問題である。この特定の力学に対する非形式的な議論は、密な区分写像、偶奇に依存する区間、複雑な行間遷移を含む約 20 ページに及ぶ。この規模の形式検証に取り組むため LEAP を導入したところ、単一の巨大な非形式証明を、細粒度で高度に構造化された証明グラフへ分解できた。LEAP はグラフ内で相互依存するノードを自律的かつ体系的に解決し、複雑なサイクル結合の力学を完全に検証した。最終的に 5000 行を超える厳密な Lean 4 コードを合成し、この部分問題の形式証明を完成させた。完全な問題記述と非形式証明は [https://github.com/dpwoodru/knuthCycles/tree/main](https://github.com/dpwoodru/knuthCycles/tree/main) で公開している。

**Erdős 問題 457 の形式化。** さらに、三角形を含まないグラフの密度に関する古典的なグラフ理論問題、Erdős 問題 457 で LEAP を試した。この問題はすでに解決済みだが、確立した数学的結果を LEAP が自律的に再構成し検証する能力を評価する格好のベンチマークとなった。Lean 4 で既知の証明を第一原理から導出する課題を与えると、LEAP は組合せ論的制約を適切に処理し、定理の妥当性を確認した。この再現の成功は、LEAP が人間の介入なしに、複雑な既存文献を高保証の形式証明へ確実に翻訳できることを示す。

形式文と詳しい問題記述は[第 9 節](#section-9)に示す。

<span id="section-7"></span>

## 7 結論と今後の課題

LEAP の成功は、適切な構造的足場と組み合わせれば、現代の汎用 LLM が厳密な分野固有タスクに対して十分な推論能力を持つことを示唆する。形式数学では、この足場は自然に証明分解と検証器誘導型の改善という形を取る。モデルが複雑な定理を小さなサブゴールへ分解し、Lean コンパイラが各形式段階を検査する。この設計は、非形式推論を機械的に検証された証明へ翻訳する構造化された仕組みを与える。
今後の研究における中心的な課題は、得られた証明木を効率よくたどる方法である。分解がより細粒度のサブゴールを生むにつれて、探索空間は急速に拡大し得る。したがって今後のシステムは、大規模な証明探索における分岐の優先順位付け、分解戦略、計算資源の配分を改善すべきである。エージェント型形式証明システムをより複雑な数学問題へ拡張するには、このような進歩が不可欠となる。

## 謝辞

Knuth の Cycles 問題の偶数の場合に対する非形式証明を提供してくれた Michael P. Brenner、Honghao Lin、David Woodruff、Vahab Mirrokni に感謝する。また、Lean-IMO-Bench の Lean 問題文を形式化した Ashley Aragorn Khoo、Paul Lezeau、Calle Sönne、Moritz Firsching にも感謝する。

<span id="section-8"></span>

## 8 関連研究

**ニューラル定理証明** 初期のニューラル定理証明研究は、Metamath [Sut20]、MM0 [Car20]、または幾何問題専用の形式言語 [Lu21a] など、主として内製の記号エンジンを利用した。
mathlib [Mat20b]、LeanDojo [Yan23e]、MiniF2F [Zhe22a] などの後続研究は、Lean での生成的定理証明に LLM を用いる先駆けとなった。
これらは、既知の定理から成る豊富なライブラリ、段階単位の探索を行う対話的環境、適度な難度の評価集合を提供する柱となっている。
大きな探索空間を扱うため、HyperTree Proof Search [Lam22] と関連する Monte Carlo 木探索手法 [Lin25a, Xin25b] が検討されてきた。
探索ベースの手法が tactic 水準で動作する一方、Baldur [Fir23] と DeepSeek-prover-v1.5 [Xin25] は、完全な証明を一度に生成しようとする証明全体の生成を検討した。非形式証明やスケッチで形式証明探索を導くことも有望な方向である。「draft, sketch, and prove」手法 [Jia23d] は、非形式証明をブループリントとして使うことで、形式定理証明を大きく導き改善できると示した。LEAP はこの発想を発展させ、汎用 LLM で非形式ブループリントを生成し、コンパイラのフィードバックに基づいて形式証明を反復的に改善するが、形式化段階で専用の微調整済みモデルには依存しない。

**専用証明器モデル** 近年の画期的成果は、形式数学コーパスで大規模モデルを広範に微調整することに依存する場合が多い。代表的な研究に AlphaProof [Hub26]、DeepSeek Prover V2 [Ren25b]、Seed Prover [Che25h]、Kimina Prover [Wan25l]、Goedel Prover V2 [Lin25j] がある。これらのモデルは、形式システム上の訓練と探索を大規模化して最先端の性能を達成する。しかし訓練に多量の計算資源が必要で、特定の形式言語に強く特化している。対照的に LEAP は、汎用 LLM を適切なエージェント環境へ置けば、このような専用微調整なしでも競争力のある性能を達成できると示す。

**自動形式化** 自然言語の数学を形式文と証明へ翻訳する自動形式化は、非形式推論と形式推論を結ぶ重要な橋である。初期研究はニューラル機械翻訳技術に依存していた [Wu22a]。近年は、AlphaProof [Hub26] の自動形式化パイプラインに見られるように、証明器を大規模に訓練する形式文の生成に LLM が使われている。LEAP はエージェント型ハーネス内で汎用 LLM の高い自動形式化能力を利用し、非形式ブループリントと形式証明の隔たりを埋める。

**LLM による数学推論** 大規模言語モデルは自然言語の数学問題を解くうえで目覚ましく進歩し、複雑な推論に高い能力を示してきた。OpenAI o1 [Ope24h] や DeepSeek R1 [Dee25c] などの近年の進展は、複雑な数学タスクで強化学習を大規模化する有効性を示し、AIME のような競技ベンチマークで高得点を達成している。しかし、これらのモデルを形式定理証明ベンチマークで直接評価すると成功率が低い場合が多く、非形式推論と形式検証の隔たりが明らかになる。LEAP は、エージェント型ハーネス内で汎用 LLM の高い非形式推論能力と指示追従能力を利用し、Lean コンパイラとのやり取りと反復的な自己修正を可能にすることで、専用微調整なしに形式化の隔たりを埋める。

<span id="section-9"></span>

## 9 問題文

LEAP で試した未解決問題の LEAN 文を示す。

**有向 Cayley グラフの Hamilton 分解** 有向 Cayley グラフ $\Gamma_{m}=Cay(\mathbb{Z}_{m}^{3},\{e_{1},e_{2},e_{3}\})$ の Hamilton 分解問題は、その辺を互いに異なる有向 Hamilton サイクル 3 本へ分割できるかを問う。偶数の場合の構成（$m=2h \ge 10$）では、各色クラスの三次元ルーティング力学を、$\mathbb{Z}_{m} \times \mathbb{Z}_{m}$ 格子上で定義した二次元平面の「round map」へ解析的に射影できる。次の形式文は、偶奇に依存する構造的欠陥、座標シフト、区分的遷移を含め、Color 2 部分グラフの厳密な操作的意味論を符号化し、その round map が長さ $m^2$ の単一で途切れないサイクルになると主張する。

**有向 Cayley グラフの Hamilton 分解に対する Lean 文**

```lean
import Mathlib

set_option autoImplicit false

variable (h : ℕ) (hh : 5 ≤ h)

abbrev Fiber2 (h : ℕ) := Fin (2 * h) × Fin (2 * h)

-- 1. Base Coordinate Definitions
def one2 : Fin (2 * h) := ⟨1, by omega⟩
def mMinusOne2 : Fin (2 * h) := ⟨2 * h - 1, by omega⟩
def mMinusTwo2 : Fin (2 * h) := ⟨2 * h - 2, by omega⟩

def succ2c (x : Fin (2 * h)) : Fin (2 * h) := x + one2 h hh
def pred2c (x : Fin (2 * h)) : Fin (2 * h) := x - one2 h hh

-- 2. Exceptional Set Logic (Defects)
def y2SwitchRow (x : Fin (2 * h)) : Prop :=
  x.val = h + 1 ∨ x.val = h + 2 ∨ x.val = h + 3

instance (x : Fin (2 * h)) : Decidable (y2SwitchRow h x) := by
  unfold y2SwitchRow
  infer_instance

def y2star (x : Fin (2 * h)) : Fin (2 * h) :=
  if y2SwitchRow h x then
    if h % 2 = 0 then mMinusTwo2 h hh else mMinusOne2 h hh
  else
    ⟨2 * h - 1 - x.val, by omega⟩

def A2 (x : Fin (2 * h)) : Fin (2 * h) :=
  succ2c h hh (y2star h hh x)

def activeB2 (x y : Fin (2 * h)) : Prop :=
  if h % 2 = 0 then
    (x.val = h + 1 ∧ y.val ≤ h - 1) ∨
      (x.val = h + 4 ∧ h - 3 ≤ y.val ∧ y.val ≤ 2 * h - 2)
  else
    (x.val = h + 1 ∧ 1 ≤ y.val ∧ y.val ≤ h - 1) ∨
      (x.val = h + 4 ∧ h - 3 ≤ y.val)

instance (x y : Fin (2 * h)) : Decidable (activeB2 h x y) := by
  unfold activeB2
  infer_instance

-- 3. The Round Map
def r2Map (p : Fiber2 h) : Fiber2 h :=
  let x := p.1
  let u := pred2c h hh p.2
  if u = A2 h hh x then
    (succ2c h hh x,
      if x.val = h + 1 ∨ x.val = h + 2 then u else pred2c h hh u)
  else if activeB2 h x u then
    (x, pred2c h hh u)
  else
    (x, u)

-- 4. The Self-Contained Goal
/-- The unrolled Hamiltonicity goal for the Color 2 round map. -/
theorem color2_singleCycle_unrolled (h6 : 6 ≤ h) :
    (∀ p : Fin (2 * h) × Fin (2 * h), (r2Map h hh)^[(2 * h) * (2 * h)] p = p) ∧
    (∀ (p : Fin (2 * h) × Fin (2 * h)) (k : ℕ), 0 < k → k < (2 * h) * (2 * h) → (r2Map h hh)^[k] p ≠ p) := by
  sorry
```

**Erdős 457** Erdős 問題 457 は、連続する整数の素因数に関する数論の課題である。具体的には、ある実数 $\varepsilon > 0$ が存在し、無限に多くの整数 $n$ について、$p \le (2 + \varepsilon)\log n$ を満たすすべての素数 $p$ が、$n+1$ から始まる $\lfloor\log n\rfloor$ 個の連続整数の積を割り切ると予想する。次の Lean 形式化は、この漸近的な素数の可除性条件を正確に表す。

**Erdős 問題 457 に対する Lean 文**

```lean
import Mathlib

theorem erdos_457 : ∃ ε > (0 : ℝ),
    { (n : ℕ) | ∀ (p : ℕ), p ≤ (2 + ε) * Real.log n → p.Prime →
      p ∣ ∏ i ∈ Finset.Icc 1 ⌊Real.log n⌋₊, (n + i) }.Infinite := by
  sorry
```

<span id="section-10"></span>

## 10 証明コンテキストと成果物

本節では、LEAP が証明計画中に用いる形式的および非形式的な成果物を説明する。形式成果物は、コンパイラが検査するか証明 DAG で表現される Lean 水準のオブジェクトに対応し、非形式成果物は直接証明と分解を導く自然言語の計画オブジェクトに対応する。

**形式コンテキスト。** *証明ゴール*とは、まだ証明されていない Lean の定理文または補題文である。元の入力定理が根の証明ゴールとなり、分解で導入した補題文が証明 DAG のサブゴールになる。*形式証明*とは、新たに提案した未証明の補題に依存しない、現在の証明ゴールに対する完全な Lean 証明である。Lean コンパイラに受理されると、対応するゴールは解決済みと記録される。*証明スケッチ*とは、提案補題の集合を仮定して現在のゴールを証明する Lean 成果物である。LEAP では、明示的に提案した補題に限り、証明スケッチに `sorry` プレースホルダを含められる。したがって、検証済み証明スケッチは有効な分解を定義する。後で参照するすべての提案補題が証明されれば、現在のゴールも証明される。Lean-IMO-Bench の Basic 集合にある問題 001 と 009 を用いて、これらのコンテキストの例を示す。<strong>根問題の証明ゴールを除くすべての成果物は、LEAP が自動的に作成する。）</strong>

**証明ゴールの例（Lean-IMO-Bench、Basic 001）**

```lean
theorem PBBasic001 : {f : ℤ → ℤ | ∀ x y, f (2 * x) + 2 * f y = f (f (x + y))}
  = {0} ∪ {(fun x ↦ 2 * x + c)| (c : ℤ)} :=
by sorry
```

**形式証明の例（Lean-IMO-Bench、Basic 001）**

```lean
import Mathlib

theorem PBBasic001 : {f : ℤ → ℤ | ∀ x y, f (2 * x) + 2 * f y = f (f (x + y))}
      = {0} ∪ {(fun x ↦ 2 * x + c)| (c : ℤ)} :=
by
  ext f
  simp only [Set.mem_setOf_eq, Set.mem_union, Set.mem_singleton_iff, Set.mem_range]
  constructor
  · intro h
    have h1 : ∀ y, f (f y) = 2 * f y + f 0 := by
      -- [Proof details omitted for brevity]
    have h2 : ∀ x, f (2 * x) = 2 * f x - f 0 := by
      -- [Proof details omitted]
    have h3 : ∀ x y, f (x + y) = f x + f y - f 0 := by
      -- [Proof details omitted]

    have hc : ∃ c, c = f 0 := ⟨f 0, rfl⟩
    rcases hc with ⟨c, hc_eq⟩
    have hk : ∃ k, k = f 1 - c := ⟨f 1 - c, rfl⟩
    rcases hk with ⟨k, hk_eq⟩

    -- ... [Induction steps for h4 omitted] ...

    have h5 : ∀ x : ℤ, f x = k * x + c := by
      -- ... [Negative cases to prove linear form omitted] ...

    have eq_all : ∀ x y : ℤ, k * (2 * x) + c + 2 * (k * y + c) = k * (k * (x + y) + c) + c := by
      intro x y
      have h_orig := h x y
      simp only [h5] at h_orig
      exact h_orig

    have hk_eq : k * (k - 2) = 0 := by
      -- [Algebraic simplification using eq_all 1 0 and eq_all 0 0 omitted]

    have hk2 : k = 0 ∨ k = 2 := by
      cases mul_eq_zero.mp hk_eq with
      | inl h1 => left; exact h1
      | inr h2 => right; omega

    rcases hk2 with hk0 | hk2
    · left
      have eq00 := eq_all 0 0
      rw [hk0] at eq00
      have hc0 : c = 0 := by linarith [eq00]
      ext x
      simp only [Pi.zero_apply]
      have hfx := h5 x
      rw [hk0, hc0] at hfx
      omega
    · right
      use c
      ext x
      have hfx := h5 x
      rw [hk2] at hfx
      omega

  -- Prove that the derived candidates are indeed solutions mappings
  · rintro (rfl | ⟨c, rfl⟩)
    · intro x y
      simp only [Pi.zero_apply, mul_zero, add_zero]
    · intro x y
      dsimp only
      ring
```

**証明スケッチの例（Lean-IMO-Bench、Basic 006）**

```lean
import Mathlib

open Polynomial

-- Shared Definitions from the file environment
def K (c : ℕ → ℤ) : ℤ := (c 1)^2 - (2 : ℤ) * (c 0) * (c 2)

def k_target (c : ℕ → ℤ) : ℕ := max 2 (Int.toNat (K c + (1 : ℤ)))

def esymm_one_target (s : Multiset ℝ) : ℕ → ℝ
  | 0 => 0
  | k + 1 => s.esymm k

def esymm_two_target (s : Multiset ℝ) : ℕ → ℝ
  | 0 => 0
  | 1 => 0
  | k + 2 => s.esymm k

-- Supporting Lemmas (with sorry)
lemma root_count_bound_implies_eq (c : ℕ → ℤ) (hc : c 0 ≠ 0) (k : ℕ)
  (h_not_less : ¬ (((∑ i ∈ Finset.Icc 0 k, monomial i (c i)).rootSet ℝ).ncard < k)) :
  ((∑ i ∈ Finset.Icc 0 k, monomial i (c i)).rootSet ℝ).ncard = k ∧
  (∑ i ∈ Finset.Icc 0 k, monomial i (c i)).natDegree = k := by
  sorry

lemma k_le_K_of_eq (c : ℕ → ℤ) (hc : c 0 ≠ 0) (k : ℕ) (hk : (2 : ℕ) ≤ k)
  (h_eq : ((∑ i ∈ Finset.Icc 0 k, monomial i (c i)).rootSet ℝ).ncard = k)
  (h_deg : (∑ i ∈ Finset.Icc 0 k, monomial i (c i)).natDegree = k) :
  (k : ℤ) ≤ K c := by
  sorry

lemma k_target_ge_two (c : ℕ → ℤ) : (2 : ℕ) ≤ k_target c := by
  sorry

lemma k_target_gt_K (c : ℕ → ℤ) : K c < (k_target c : ℤ) := by
  sorry

-- Main Theorem
theorem PBBasic006 (c : ℕ → ℤ) (hc : c 0 ≠ 0) :
  ∃ k, ((∑ i ∈ Finset.Icc 0 k, monomial i (c i)).rootSet ℝ).ncard < k := by
  use k_target c
  by_contra h
  have h1 := root_count_bound_implies_eq c hc (k_target c) h
  have h2 := k_le_K_of_eq c hc (k_target c) (k_target_ge_two c) h1.1 h1.2
  have h3 := k_target_gt_K c
  linarith
```

**非形式コンテキスト。** *非形式証明*とは、新しいサブゴールへの分解を導入せず、現在のゴールを直接証明するための自然言語による計画である。これは完全な形式証明の生成を導く。*非形式ブループリント*とは、現在のゴールを有用な補助補題へ帰着する方法を説明する、より高水準の自然言語計画である。非形式証明とは異なり、非形式ブループリントはすぐには証明しない提案補題を導入でき、後の証明段階に使う補助補題も含められる。これらの提案補題は形式サブゴールへ翻訳され、証明 DAG 内で構成される。以下に例を示す。

<strong>非形式証明の例（Lean-IMO-Bench、Basic 001）</strong>

$S = \{f : \mathbb{Z} \to \mathbb{Z} \mid \forall x, y \in \mathbb{Z}, f(2x) + 2f(y) = f(f(x+y))\}$ および $T = \{0\} \cup \{x \mapsto 2x + c \mid c \in \mathbb{Z}\}$ とする。$S = T$ を示したい。

<strong>第 1 部：$T \subseteq S$ の証明</strong>

$f \in T$ と仮定する。このとき $f$ は定数零関数（$f = 0$）であるか、ある $c \in \mathbb{Z}$ に対して $f(x) = 2x + c$ である。

- 場合 1：$f = 0$ なら、任意の $x, y \in \mathbb{Z}$ に対し、左辺は $f(2x) + 2f(y) = 0 + 2(0) = 0$ である。右辺は $f(f(x+y)) = f(0) = 0$ である。両辺は一致するため、$f \in S$ となる。
- 場合 2：ある $c \in \mathbb{Z}$ に対して $f(x) = 2x + c$ なら、任意の $x, y \in \mathbb{Z}$ に対して左辺は次のようになる。
  $f(2x) + 2f(y) = (2(2x) + c) + 2(2y + c) = 4x + 4y + 3c$.
  右辺は次のとおりである。
  $f(f(x+y)) = f(2(x+y) + c) = 2(2x + 2y + c) + c = 4x + 4y + 3c$.
  両辺は等しいため、$f \in S$ となる。

以上で $T \subseteq S$ が示された。

<strong>第 2 部：$S \subseteq T$ の証明</strong>

$f \in S$ とする。このとき、すべての $x, y \in \mathbb{Z}$ に対して次が成り立つ。

$(1)\ f(2x) + 2f(y) = f(f(x+y))$.

式（1）に $x = 0$ を代入する。
$f(0) + 2f(y) = f(f(y))$.
$c = f(0)$ とする。このとき、すべての $y \in \mathbb{Z}$ に対して次が成り立つ。

$(2)\ f(f(y)) = 2f(y) + c$.

式（2）は $x+y$ を含む任意の入力について成り立つため、式（1）の右辺に用いると次を得る。

$(3)\ f(2x) + 2f(y) = 2f(x+y) + c$.

式（3）に $y = 0$ を代入する。
$f(2x) + 2f(0) = 2f(x) + c$.
$f(0) = c$ なので、これは $f(2x) + 2c = 2f(x) + c$ となり、次を得る。

$(4)\ f(2x) = 2f(x) - c$.

ここで式（4）を式（3）の左辺へ代入する。
$(2f(x) - c) + 2f(y) = 2f(x+y) + c$.
項を整理すると次を得る。
$2f(x+y) = 2f(x) + 2f(y) - 2c$.
終域は $\mathbb{Z}$ なので、両辺を $2$ で割って次を得る。

$(5)\ f(x+y) = f(x) + f(y) - c$.

$g(x) = f(x) - c$ によって新しい関数 $g: \mathbb{Z} \to \mathbb{Z}$ を定義する。このとき式（5）は次のように書き換えられる。
$g(x+y) + c = (g(x) + c) + (g(y) + c) - c$.
簡単化すると Cauchy の関数方程式を得る。
すべての $x, y \in \mathbb{Z}$ に対して $g(x+y) = g(x) + g(y)$ である。

$g$ は整数上で $g(x+y) = g(x) + g(y)$ を満たすため、線形関数でなければならない。$k = g(1) \in \mathbb{Z}$ とする。標準的な帰納法により、すべての整数 $x \ge 0$ に対して $g(x) = kx$ となる（$g(0) = 0$）。負の整数については、$0 = g(0) = g(x + (-x)) = g(x) + g(-x)$ なので、$g(x) = -g(-x) = -(-kx) = kx$ となる。したがって、すべての $x \in \mathbb{Z}$ に対して $g(x) = kx$ である。
その結果、すべての $x \in \mathbb{Z}$ に対して $f(x) = kx + c$ を得る。

$k$ と $c$ に許される値を求めるため、$f(x) = kx + c$ を元の式（1）へ代入する。
左辺：$f(2x) + 2f(y) = k(2x) + c + 2(ky + c) = 2kx + 2ky + 3c$。
右辺：$f(f(x+y)) = k(f(x+y)) + c = k(k(x+y) + c) + c = k^2x + k^2y + (k+1)c$。

等式 $2kx + 2ky + 3c = k^2x + k^2y + (k+1)c$ がすべての $x, y \in \mathbb{Z}$ に対して成り立つには、対応する係数が一致しなければならない。
$x=0$、$y=0$ とすると次を得る。

$(6)\ 3c = (k+1)c$.

$x=1$、$y=0$ とすると次を得る。
$2k + 3c = k^2 + (k+1)c$.
この式から式（6）を引くと $2k = k^2$ となり、$k(k-2) = 0$ と簡単化できる。$k \in \mathbb{Z}$ なので、解は $k = 0$ または $k = 2$ だけである。

- 場合 A：$k = 0$。
  式（6）へ $k = 0$ を代入する。
  $3c = c \implies 2c = 0 \implies c = 0$.
  したがって $f(x) = 0x + 0 = 0$ であり、$f$ は零関数である。ゆえに $f \in \{0\} \subseteq T$ となる。
- 場合 B：$k = 2$。
  式（6）へ $k = 2$ を代入する。
  $3c = 3c$ は、任意の $c \in \mathbb{Z}$ について成り立つ。
  したがって、ある $c \in \mathbb{Z}$ に対して $f(x) = 2x + c$ である。ゆえに $f \in \{x \mapsto 2x + c \mid c \in \mathbb{Z}\} \subseteq T$ となる。

いずれの場合も、任意の関数 $f \in S$ は $T$ にも含まれるため、$S \subseteq T$ が示された。
両方向の包含関係が成り立ったので、$S = T$ である。

<strong>非形式ブループリントの例（Lean-IMO-Bench、Basic 006）</strong>

<strong>高水準の数学的アイデア</strong>

整数係数多項式 $P(x) = \sum_{i=0}^k c_i x^i$ の次数が $k$ で、実根がちょうど $k$ 個あるとき、$K(c) = c_1^2 - 2c_0 c_2$ として $k \le K(c)$ を証明することが目標である。

この証明は多重集合対称多項式と AM-GM 不等式を巧みに用い、有理関数と多項式の微分を避ける。

1. **多項式の根と分解**：$P$ の自然次数は $k$ で、相異なる実根を $k$ 個持つため、$\mathbb{R}$ 上で完全に分解し、その根の多重集合 $S$ の濃度は $k$ である。
2. **Vieta の公式**：Vieta の公式により、係数 $c_0, c_1, c_2$ を $S$ の基本対称多項式で表す。
  $c_0 = c_k (-1)^k E_k$
  $c_1 = c_k (-1)^{k-1} E_{k-1}$
  $c_2 = c_k (-1)^{k-2} E_{k-2}$
  ここで $E_i = \mathrm{esymm}_i(S)$ である。
3. **多重集合の恒等式**：各根 $x \in S$ を他のすべての根の積 $(S \setminus \{x\})$.`prod` へ写し、新しい多重集合 $Y$ を構成する。多重集合だけを用いた帰納的恒等式から次が分かる。
  $\sum Y = E_{k-1}$
  $\mathrm{esymm}_2(Y) = E_k E_{k-2}$
  $\prod Y = (E_k)^{k-1}$
4. **平方和**：多重集合 $Z = c_k Y$ について、その要素の平方から成る $W = \{ z^2 \mid z \in Z \}$ の和を評価する。
  関係 $(\sum Z)^2 = \sum (Z^2) + 2 \mathrm{esymm}_2(Z)$ により、$\sum W = c_1^2 - 2c_0 c_2 = K(c)$ を代数的に計算できる。
5. **整数積の下界**：$Z$ の積は $c_k ((-1)^k c_0)^{k-1}$ となる。$P$ の次数は $k$ で $c_0 \neq 0$ なので、$c_0$ と $c_k$ はともに非零整数である。したがって $Z$ の積は非零整数であり、$\prod W = (\prod Z)^2 \ge 1$ を得る。
6. **AM-GM 不等式**：多重集合 $W$（積が $\ge 1$ である $k$ 個の非負実数から成る）に AM-GM 不等式を適用し、$\sum W \ge k \implies K(c) \ge k$ を得る。

<strong>必要な大域的定義、変数、構造</strong>

新しい定義、公理、構造は不要である。`Multiset`、`Polynomial`、`esymm` など、標準の Mathlib コンポーネントだけを用いる。

**小さな補題（部分問題）**

```lean
lemma coeff_of_sum_Icc (c : ℕ → ℤ) (k : ℕ) (hk : (2 : ℕ) ≤ k) (i : ℕ) (hi : i ≤ k) :
  (∑ j ∈ Finset.Icc 0 k, Polynomial.monomial j (c j)).coeff i = c i
```

**目的**：与えられた和の形式から多項式の係数を取り出す処理を簡単にする。

```lean
lemma card_roots_eq_of_ncard_rootSet {k : ℕ} {P : Polynomial ℝ}
  (h_deg : P.natDegree = k)
  (h_ncard : (P.rootSet ℝ).ncard = k) :
  P.roots.card = k
```

**目的**：次数 $k$ で相異なる実根を $k$ 個持つ多項式は、重複度を数えても根がちょうど $k$ 個であると示す。

```lean
lemma multiset_map_erase_prod_sum {R : Type*} [CommRing R] (s : Multiset R) :
  (s.map (fun x => (s.erase x).prod)).sum = s.esymm (s.card - 1)
```

**目的**：各要素を除いた積の和を $E_{k-1}$ と結び付ける多重集合の恒等式。

```lean
lemma multiset_map_erase_prod_esymm_two {R : Type*} [CommRing R] (s : Multiset R) :
  (s.map (fun x => (s.erase x).prod)).esymm 2 = s.prod * s.esymm (s.card - 2)
```

**目的**：要素を除いた積の第 2 基本対称多項式を $E_k E_{k-2}$ と結び付ける多重集合の恒等式。

```lean
lemma multiset_sum_sq_eq {R : Type*} [CommRing R] (s : Multiset R) :
  (s.map (fun x => x^2)).sum = (s.sum)^2 - (2 : R) * s.esymm 2
```

**目的**：多重集合の平方和を、その和と第 2 基本対称多項式によって表す。

```lean
lemma multiset_map_erase_prod_prod {R : Type*} [CommRing R] (s : Multiset R) :
  (s.map (fun x => (s.erase x).prod)).prod = s.prod ^ (s.card - 1)
```

**目的**：要素を除いた積から成る多重集合の全要素の積を計算する。

```lean
lemma multiset_sum_ge_card_of_prod_ge_one (W : Multiset ℝ) (hw : ∀ x ∈ W, 0 ≤ x) (hp : (1 : ℝ) ≤ W.prod) :
  (W.card : ℝ) ≤ W.sum
```

**目的**：積が 1 以上の多重集合に特化した AM-GM 不等式であり、その和が濃度を下界に持つと証明する。

**証明本体の概要**

1. $P$ を和 $\sum_{i \in \texttt{Finset.Icc}\  0 k} \texttt{monomial}\  i (c_i)$ として、$P_R$ を $P.\texttt{map}\  (\texttt{algebraMap}\  \mathbb{Z} \text{ } \mathbb{R})$ として定義する。
2. `coeff_of_sum_Icc` を適用し、$i \in \{0, 1, 2, k\}$ に対して $P_R.\texttt{coeff}\  i = (c_i : \mathbb{R})$ と示す。
3. `card_roots_eq_of_ncard_rootSet` と自然次数の単射性を用いて $P_R.\texttt{roots.card} = k$ を確立する。
4. `Polynomial.splits_iff_card_roots` から $P_R.\texttt{splits}\  (\texttt{RingHom.id}\  \mathbb{R})$ が従うことを確立する。
5. $s = P_R.\texttt{roots}$ とする。Vieta の公式（`Polynomial.coeff_eq_esymm_roots_of_splits`）を呼び出し、$c_0$、$c_1$、$c_2$ を $s.\texttt{esymm}\  i$ で表す。
6. 理論的ブループリントに対応する多重集合 $Y$ と $Z$ を定義する。`multiset_sum_sq_eq`、`multiset_map_erase_prod_sum`、`multiset_map_erase_prod_esymm_two` を用い、$Z$ の要素の平方和を代数的に展開すると、ちょうど $(c_1^2 - 2c_0 c_2 : \mathbb{R}) = (K(c) : \mathbb{R})$ になると示す。
7. `multiset_map_erase_prod_prod` を用いて $Z.\texttt{prod} = c_k ((-1)^k c_0)^{k-1}$ を求める。
8. $c_0$ と $c_k$ は非零整数なので、その代数的な組合せ $Z.\texttt{prod}$ は非零整数を表す。したがって、その平方（$W = Z^2$ の積）は $\ge 1$ である。
9. $W$ を `multiset_sum_ge_card_of_prod_ge_one` へ渡し、$W.\texttt{sum} \ge W.\texttt{card}$ を導く。
10. $W.\texttt{card} = k$ と $W.\texttt{sum} = (K(c) : \mathbb{R})$ を用いて、$(k : \mathbb{R}) \le (K(c) : \mathbb{R})$ を導く。`norm_cast` を使い、これを $(k : \mathbb{Z}) \le K(c)$ へ戻す。
