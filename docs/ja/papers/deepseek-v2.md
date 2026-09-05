---
title: 'DeepSeek-V2'
createTime: 2026/09/05 18:30:00
permalink: /ja/papers/deepseek-v2/
pageClass: paper-reading
---

> [DeepSeek-AI](https://www.deepseek.com/)。2024 年 5 月 7 日に arXiv へ初投稿されました。現在のバージョンは v5（2024 年 6 月 19 日）です。[DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)。[原論文 PDF](/paper/deepseek-v2.pdf)。[DOI](https://doi.org/10.48550/arXiv.2405.04434)。[TeX ソース](https://export.arxiv.org/e-print/2405.04434v5)。正確な印刷レイアウトと参考文献については、元の PDF を正本とします。

## 概要

本稿では、学習コストが経済的で推論効率に優れた強力な Mixture-of-Experts（MoE）言語モデルである DeepSeek-V2 を提示します。総パラメータは 236B で、各トークンにつきそのうち 21B を活性化し、128K トークンのコンテキスト長をサポートします。DeepSeek-V2 は、Multi-head Latent Attention（MLA）と DeepSeekMoE を含む革新的なアーキテクチャを採用しています。MLA は Key-Value（KV）キャッシュを潜在ベクトルへ大幅に圧縮することで効率的な推論を保証し、DeepSeekMoE はスパース計算によって経済的なコストで強力なモデルを学習できるようにします。DeepSeek 67B と比べ、DeepSeek-V2 は著しく強い性能を達成しつつ、学習コストを 42.5% 節約し、KV キャッシュを 93.3% 削減し、最大生成スループットを 5.76 倍に高めます。DeepSeek-V2 を 8.1T トークンからなる高品質かつ複数ソースのコーパスで事前学習し、さらに Supervised Fine-Tuning（SFT）と強化学習（RL）を施してその潜在能力を最大限に引き出します。評価結果は、わずか 21B の活性化パラメータしか持たないにもかかわらず、DeepSeek-V2 とそのチャット版がオープンソースモデルの中でトップクラスの性能を達成することを示しています。モデルのチェックポイントは https://github.com/deepseek-ai/DeepSeek-V2 で公開しています。

<span id="figure-01"></span>

![DeepSeek-V2 図 1](../../papers/deepseek-v2/figure-01.png)

**図 1.** (a) さまざまなオープンソースモデルにおける MMLU 精度と活性化パラメータの関係。 (b) DeepSeek 67B（Dense）と DeepSeek-V2 の学習コストと推論効率。

<span id="section-1"></span>

## 1 はじめに

ここ数年、大規模言語モデル（LLM）[Ope22, Ope23, Ant23, Goo23] は急速に発展し、汎用人工知能（AGI）の夜明けを垣間見せてくれています。一般に、LLM の知能はパラメータ数の増加に伴って向上する傾向があり、さまざまなタスクで創発的な能力を発揮できるようになります [Wei22d]。しかし、この向上は学習時の大規模な計算リソースと、推論スループットの潜在的な低下という代償を伴います。これらの制約は、LLM の広範な採用と利用を妨げる重大な課題をもたらします。この問題に取り組むため、革新的な Transformer アーキテクチャによって経済的な学習と効率的な推論を実現する強力なオープンソースの Mixture-of-Experts（MoE）言語モデル、DeepSeek-V2 を発表します。総パラメータは 236B で、各トークンにつきそのうち 21B を活性化し、128K トークンのコンテキスト長をサポートします。

Transformer フレームワーク [Vas17d] 内のアテンションモジュールと Feed-Forward Networks（FFN）を、提案する **Multi-head Latent Attention（MLA）** と **DeepSeekMoE** によって最適化します。

1. アテンション機構の文脈では、Multi-Head Attention（MHA）[Vas17d] の Key-Value（KV）キャッシュが LLM の推論効率に対する大きな障害となります。この問題に対処するため、Grouped-Query Attention（GQA）[Ain23] や Multi-Query Attention（MQA）[Sha19] を含むさまざまな手法が探求されてきました。しかし、これらの手法は KV キャッシュを減らそうとする際に性能を犠牲にすることがよくあります。両方の利点を兼ね備えるため、低ランクの鍵・値同時圧縮を備えたアテンション機構である MLA を導入します。実験的に、MLA は MHA より優れた性能を達成し、同時に推論中の KV キャッシュを大幅に削減して推論効率を高めます。
2. Feed-Forward Networks（FFN）については、エキスパート専門化の可能性を高めるため、細粒度のエキスパート分割と共有エキスパート分離を採用する DeepSeekMoE アーキテクチャ [Dai24] に従います。DeepSeekMoE アーキテクチャは GShard [Lep20] のような従来の MoE アーキテクチャと比べて大きな利点を示し、経済的なコストで強力なモデルを学習できるようにします。学習中にエキスパート並列を採用するため、通信オーバーヘッドを制御し負荷バランスを確保する補助的な仕組みも考案しました。これら二つの技法を組み合わせることで、DeepSeek-V2 は強力な性能（[図 1](#figure-01)(a)）、経済的な学習コスト、効率的な推論スループット（[図 1](#figure-01)(b)）を同時に実現します。

<span id="figure-02"></span>

![DeepSeek-V2 図 2](../../papers/deepseek-v2/figure-02.png)

**図 2.** DeepSeek-V2 のアーキテクチャの図解。MLA は生成時の KV キャッシュを大幅に削減することで効率的な推論を保証し、DeepSeekMoE はスパースなアーキテクチャを通じて経済的なコストで強力なモデルの学習を可能にします。

8.1T トークンからなる高品質かつ複数ソースの事前学習コーパスを構築しました。DeepSeek 67B（以前のリリース）[Dee24e] で使用したコーパスと比べ、このコーパスは特に中国語データの量が増え、データ品質も高くなっています。まず完全な事前学習コーパスで DeepSeek-V2 を事前学習します。次に、数学、コード、執筆、推論、安全性などさまざまな領域を網羅する 150 万の対話セッションを収集し、DeepSeek-V2 Chat（SFT）向けに Supervised Fine-Tuning（SFT）を実施します。最後に、DeepSeekMath [Sha24d] に従い、Group Relative Policy Optimization（GRPO）を用いてモデルを人間の好みに合わせて調整し、DeepSeek-V2 Chat（RL）を生成します。

英語と中国語の幅広いベンチマークで DeepSeek-V2 を評価し、代表的なオープンソースモデルと比較します。評価結果は、わずか 21B の活性化パラメータでも DeepSeek-V2 がオープンソースモデルの中でトップクラスの性能を達成し、最強のオープンソース MoE 言語モデルになることを示しています。[図 1](#figure-01) は、MMLU において DeepSeek-V2 が少数の活性化パラメータだけでトップクラスの性能を達成することを強調しています。さらに、[図 1](#figure-01) に示すように、DeepSeek 67B と比べて DeepSeek-V2 は学習コストを 42.5% 節約し、KV キャッシュを 93.3% 削減し、最大生成スループットを 5.76 倍に高めます。オープンエンドのベンチマークでも DeepSeek-V2 Chat（SFT）と DeepSeek-V2 Chat（RL）を評価します。特筆すべきは、DeepSeek-V2 Chat（RL）が AlpacaEval 2.0 [Dub24a] で 38.9 の長さ制御勝率、MT-Bench [Sto23e] で 8.97 の総合スコア、AlignBench [Liu23m] で 7.91 の総合スコアを達成することです。これらの英語のオープンエンド対話評価は、DeepSeek-V2 Chat（RL）がオープンソースのチャットモデルの中でトップクラスの性能を持つことを示しています。さらに AlignBench での評価は、中国語において DeepSeek-V2 Chat（RL）がすべてのオープンソースモデルを上回り、ほとんどのクローズドソースモデルさえ凌ぐことを示しています。

MLA と DeepSeekMoE に関するさらなる研究開発を促進するため、オープンソースコミュニティ向けに、MLA と DeepSeekMoE を備えたより小型のモデル DeepSeek-V2-Lite も公開します。総パラメータは 15.7B で、各トークンにつき 2.4B を活性化します。DeepSeek-V2-Lite の詳細な説明は [第 7 節](#section-7) にあります。

本稿の残りの部分では、まず DeepSeek-V2 のモデルアーキテクチャを詳しく説明します（[第 2 節](#section-2)）。続いて、学習データ構築、ハイパーパラメータ設定、インフラストラクチャ、長いコンテキストの拡張、そしてモデル性能と効率の評価を含む事前学習の取り組みを紹介します（[第 3 節](#section-3)）。その後、Supervised Fine-Tuning（SFT）、強化学習（RL）、評価結果、その他の議論を含むアライメントへの取り組みを示します（[第 4 節](#section-4)）。最後に、結論をまとめ、DeepSeek-V2 の現在の限界を考察し、今後の展望を示します（[第 5 節](#section-5)）。

<span id="section-2"></span>

## 2 アーキテクチャ

おおむね、DeepSeek-V2 は依然として Transformer アーキテクチャ [Vas17d] を採用しており、各 Transformer ブロックはアテンションモジュールと Feed-Forward Network（FFN）から構成されます。しかし、アテンションモジュールと FFN の両方について、革新的なアーキテクチャを設計・採用しています。アテンションには、低ランクの鍵・値同時圧縮を利用して推論時の鍵・値キャッシュのボトルネックを解消し、効率的な推論を支える MLA を設計しました。FFN には、経済的なコストで強力なモデルの学習を可能にする高性能な MoE アーキテクチャである DeepSeekMoE [Dai24] を採用しています。DeepSeek-V2 のアーキテクチャの図解を [図 2](#figure-02) に示し、この節では MLA と DeepSeekMoE の詳細を紹介します。その他の細部（層正規化や FFN の活性化関数など）については、特に明記しない限り、DeepSeek-V2 は DeepSeek 67B [Dee24e] の設定に従います。

<span id="section-2-1"></span>

### 2.1 Multi-Head Latent Attention: 推論効率の向上

従来の Transformer モデルは通常 Multi-Head Attention（MHA）[Vas17d] を採用しますが、生成中はその重い Key-Value（KV）キャッシュが推論効率を制限するボトルネックになります。KV キャッシュを減らすため、Multi-Query Attention（MQA）[Sha19] と Grouped-Query Attention（GQA）[Ain23] が提案されています。これらはより小さい KV キャッシュで済みますが、性能は MHA に及びません（MHA、GQA、MQA のアブレーションは [第 9.1 節](#section-9-1) に示します）。

DeepSeek-V2 では、Multi-head Latent Attention（MLA）と呼ばれる革新的なアテンション機構を設計しました。低ランクの鍵・値同時圧縮を備えた MLA は、MHA より優れた性能を達成しつつ、大幅に少ない KV キャッシュで済みます。以下でそのアーキテクチャを紹介し、[第 9.2 節](#section-9-2) で MLA と MHA の比較も示します。

<span id="section-2-1-1"></span>

#### 2.1.1 予備知識: 標準的な Multi-Head Attention

背景として標準的な MHA 機構をまず紹介します。$d$ を埋め込み次元、$n_h$ をアテンションヘッド数、$d_h$ をヘッドごとの次元、$\mathbf{h}_{t} \in \mathbb{R}^{d}$ をアテンション層における $t$ 番目のトークンのアテンション入力とします。標準的な MHA はまず、三つの行列 $W^{Q}, W^{K}, W^{V} \in \mathbb{R}^{d_h n_h \times d}$ を通じて $\mathbf{q}_{t}, \mathbf{k}_{t}, \mathbf{v}_{t} \in \mathbb{R}^{d_h n_h}$ をそれぞれ生成します。

<span id="equation-01"></span>

$$
\begin{aligned}
    \mathbf{q}_{t} &= W^{Q} \mathbf{h}_{t}, \\
    \mathbf{k}_{t} &= W^{K} \mathbf{h}_{t}, \\
    \mathbf{v}_{t} &= W^{V} \mathbf{h}_{t},
\end{aligned}
$$

続いて、マルチヘッドアテンション計算のため、$\mathbf{q}_{t}, \mathbf{k}_{t}, \mathbf{v}_{t}$ は $n_h$ 個のヘッドに分割されます。

$$
\begin{aligned}
    [\mathbf{q}_{t, 1};&\mathbf{q}_{t, 2};...;\mathbf{q}_{t, n_{h}}] = \mathbf{q}_{t}, \\
    [\mathbf{k}_{t, 1};&\mathbf{k}_{t, 2};...;\mathbf{k}_{t, n_{h}}] = \mathbf{k}_{t}, \\
    [\mathbf{v}_{t, 1};&\mathbf{v}_{t, 2};...;\mathbf{v}_{t, n_{h}}] = \mathbf{v}_{t}, \\
    \mathbf{o}_{t, i} &= \sum_{j=1}^{t} \mathop{\mathrm{Softmax}}_j(\frac{\mathbf{q}_{t, i}^\top \mathbf{k}_{j, i}}{\sqrt{d_{h}}}) \mathbf{v}_{j, i}, \\
    \mathbf{u}_{t} &= W^{O} [\mathbf{o}_{t, 1};\mathbf{o}_{t, 2};...;\mathbf{o}_{t, n_{h}}],
\end{aligned}
$$

ここで $\mathbf{q}_{t, i}, \mathbf{k}_{t, i}, \mathbf{v}_{t, i} \in \mathbb{R}^{d_h}$ はそれぞれ $i$ 番目のアテンションヘッドのクエリ、キー、値を表し、$W^{O} \in \mathbb{R}^{d \times d_h n_h}$ は出力射影行列を表します。推論中は、推論を高速化するためにすべてのキーと値をキャッシュする必要があるため、MHA は各トークンにつき $2 n_{h} d_{h} l$ 個の要素をキャッシュする必要があります。モデル展開では、この重い KV キャッシュが最大バッチサイズとシーケンス長を制限する大きなボトルネックとなります。

<span id="figure-03"></span>

![DeepSeek-V2 図 3](../../papers/deepseek-v2/figure-03.png)

**図 3.** Multi-Head Attention（MHA）、Grouped-Query Attention（GQA）、Multi-Query Attention（MQA）、Multi-head Latent Attention（MLA）の簡略図。キーと値を潜在ベクトルへ同時圧縮することで、MLA は推論中の KV キャッシュを大幅に削減します。

<span id="section-2-1-2"></span>

#### 2.1.2 低ランクの鍵・値同時圧縮

MLA の中核は、KV キャッシュを減らすためのキーと値の低ランク同時圧縮です。

<span id="equation-10"></span>

$$
\begin{aligned}
    \mathbf{c}_{t}^{\mathit{KV}} &= W^{\mathit{DKV}} \mathbf{h}_{t}, \\
    \mathbf{k}_{t}^{C} &= W^{\mathit{UK}} \mathbf{c}_{t}^{\mathit{KV}}, \\
    \mathbf{v}_{t}^{C} &= W^{\mathit{UV}} \mathbf{c}_{t}^{\mathit{KV}},
\end{aligned}
$$

ここで $\mathbf{c}_{t}^{\mathit{KV}} \in \mathbb{R}^{d_c}$ はキーと値の圧縮潜在ベクトル、$d_c (\ll d_h n_h)$ は KV 圧縮次元、$W^{\mathit{DKV}} \in \mathbb{R}^{d_c \times d}$ は下向き射影行列、$W^{\mathit{UK}},W^{\mathit{UV}} \in \mathbb{R}^{d_h n_h \times d_c}$ はキーと値の上向き射影行列です。推論中、MLA は $\mathbf{c}_{t}^{\mathit{KV}}$ だけをキャッシュすればよいため、その KV キャッシュは $d_{c}l$ 個の要素しか持たず、ここで $l$ は層数を表します。さらに推論中は、$W^{\mathit{UK}}$ を $W^{Q}$ に、$W^{\mathit{UV}}$ を $W^{O}$ に吸収できるため、アテンションのためにキーと値を計算しなくても済みます。[図 3](#figure-03) は、MLA における KV 同時圧縮がどのように KV キャッシュを減らすかを直感的に示しています。

さらに、学習中の活性化メモリを減らすため、KV キャッシュを減らせない場合でもクエリに対して低ランク圧縮を施します。

$$
\begin{aligned}
    \mathbf{c}_{t}^{Q} &= W^{\mathit{DQ}} \mathbf{h}_{t}, \\
    \mathbf{q}_{t}^{C} &= W^{\mathit{UQ}} \mathbf{c}_{t}^{Q},
\end{aligned}
$$

ここで $\mathbf{c}_{t}^{Q} \in \mathbb{R}^{d_c^{\prime}}$ はクエリの圧縮潜在ベクトル、$d_c^{\prime} (\ll d_h n_h)$ はクエリ圧縮次元、$W^{\mathit{DQ}} \in \mathbb{R}^{d_c^{\prime} \times d}, W^{\mathit{UQ}} \in \mathbb{R}^{d_h n_h \times d_c^{\prime}}$ はそれぞれクエリの下向き射影行列と上向き射影行列です。

<span id="section-2-1-3"></span>

#### 2.1.3 分離型回転位置埋め込み

DeepSeek 67B [Dee24e] に従い、DeepSeek-V2 には回転位置埋め込み（RoPE）[Su24] を用いるつもりでした。しかし、RoPE は低ランク KV 圧縮と互換性がありません。具体的には、RoPE はキーとクエリの両方に対して位置に敏感です。キー $\mathbf{k}_{t}^{C}$ に RoPE を適用すると、[式 10](#equation-10) の $W^{\mathit{UK}}$ は位置に敏感な RoPE 行列と結合されます。この場合、現在生成中のトークンに関連する RoPE 行列が $W^{Q}$ と $W^{\mathit{UK}}$ の間に位置し、行列の乗算は交換法則を満たさないため、推論中に $W^{\mathit{UK}}$ を $W^{Q}$ に吸収できなくなります。その結果、推論中にすべてのプレフィックストークンのキーを再計算しなければならず、推論効率を著しく損なうことになります。

解決策として、RoPE を運ぶために追加のマルチヘッドクエリ $\mathbf{q}_{t, i}^{R} \in \mathbb{R}^{d_h^R}$ と共有キー $\mathbf{k}_{t}^{R} \in \mathbb{R}^{d_h^R}$ を用いる分離型 RoPE 戦略を提案します。ここで $d_h^R$ は分離されたクエリとキーのヘッドごとの次元です。分離型 RoPE 戦略を備えた MLA は以下の計算を実行します。

$$
\begin{aligned}
    [\mathbf{q}_{t, 1}^{R};\mathbf{q}_{t, 2}^{R};...;\mathbf{q}_{t, n_{h}}^{R}] = \mathbf{q}_{t}^{R} &= \mathop{\mathrm{RoPE}}({W^{\mathit{QR}}} \mathbf{c}_{t}^{Q}), \\
    \mathbf{k}_{t}^{R} &= \mathop{\mathrm{RoPE}}({W^{\mathit{KR}}} \mathbf{h}_{t}), \\
    \mathbf{q}_{t, i} &= [\mathbf{q}_{t, i}^{C}; \mathbf{q}_{t, i}^{R}], \\
    \mathbf{k}_{t, i} &= [\mathbf{k}_{t, i}^{C}; \mathbf{k}_{t}^{R}], \\
    \mathbf{o}_{t, i} &= \sum_{j=1}^{t} \mathop{\mathrm{Softmax}}_j(\frac{\mathbf{q}_{t, i}^\top \mathbf{k}_{j, i}}{\sqrt{d_{h} + d_{h}^{R}}}) \mathbf{v}_{j, i}^{C}, \\
    \mathbf{u}_{t} &= W^{O} [\mathbf{o}_{t, 1};\mathbf{o}_{t, 2};...;\mathbf{o}_{t, n_{h}}],
\end{aligned}
$$

ここで $W^{\mathit{QR}} \in \mathbb{R}^{d_h^R n_h \times d_c^{\prime}}$ と $W^{\mathit{KR}} \in \mathbb{R}^{d_h^R \times d}$ はそれぞれ分離されたクエリとキーを生成する行列、$\mathop{\mathrm{RoPE}}(\cdot)$ は RoPE 行列を適用する操作、$[\cdot;\cdot]$ は連結操作を表します。推論中は分離されたキーもキャッシュする必要があります。したがって DeepSeek-V2 は $(d_{c} + d_h^R)l$ 個の要素を含む合計 KV キャッシュを必要とします。

MLA の完全な計算過程を示すため、その完全な式を [第 8 節](#section-8) にまとめて示します。

<span id="table-01"></span>

![DeepSeek-V2 表 1](../../papers/deepseek-v2/table-01.png)

**表 1.** 異なるアテンション機構間のトークンあたりの KV キャッシュ比較。$n_{h}$ はアテンションヘッド数、$d_{h}$ はアテンションヘッドごとの次元、$l$ は層数、$n_{g}$ は GQA のグループ数、$d_{c}$ と $d_h^R$ はそれぞれ MLA における KV 圧縮次元と分離されたクエリ・キーのヘッドごとの次元を表します。KV キャッシュの量は、保存精度にかかわらず要素数で測定します。DeepSeek-V2 では $d_{c}$ を $4d_{h}$、$d_h^R$ を $\frac{d_{h}}{2}$ に設定しています。したがってその KV キャッシュはわずか 2.25 グループの GQA に相当しますが、性能は MHA より優れています。

<span id="section-2-1-4"></span>

#### 2.1.4 キー・バリューキャッシュの比較

異なるアテンション機構間のトークンあたりの KV キャッシュ比較を [表 1](#table-01) に示します。MLA はわずか 2.25 グループの GQA に相当する少量の KV キャッシュしか必要としませんが、MHA より優れた性能を達成できます。

<span id="section-2-2"></span>

### 2.2 DeepSeekMoE: 経済的なコストで強力なモデルを学習する

<span id="section-2-2-1"></span>

#### 2.2.1 基本アーキテクチャ

FFN には DeepSeekMoE アーキテクチャ [Dai24] を採用します。DeepSeekMoE には二つの重要なアイデアがあります。エキスパートをより細かい粒度に分割してエキスパート専門化とより正確な知識獲得を高めること、およびルーティングされるエキスパート間の知識冗長性を軽減するために一部の共有エキスパートを分離することです。活性化エキスパートと総エキスパートパラメータが同じであれば、DeepSeekMoE は GShard [Lep20] のような従来の MoE アーキテクチャを大きく上回ることができます。

$\mathbf{u}_{t}$ を $t$ 番目のトークンの FFN 入力とすると、FFN 出力 $\mathbf{h}_{t}^{\prime}$ は次のように計算されます。

$$
\begin{aligned}
    \mathbf{h}_{t}^{\prime} & = \mathbf{u}_{t} + \sum_{i=1}^{N_{s}} {\mathop{\mathrm{FFN}}^{(s)}_{i}\left( \mathbf{u}_{t} \right)} + \sum_{i=1}^{N_r} {g_{i,t} \mathop{\mathrm{FFN}}^{(r)}_{i}\left( \mathbf{u}_{t} \right)}, \\
    g_{i,t} & = \begin{cases}
    s_{i,t}, & s_{i,t} \in \mathop{\mathrm{Topk}} (\{ s_{j, t} | 1 \leq j \leq N_r \}, K_{r}), \\
    0, & \mathrm{otherwise},
    \end{cases} \\
    s_{i,t} & = \mathop{\mathrm{Softmax}}_i \left( {\mathbf{u}_{t}}^\top \mathbf{e}_{i} \right),
\end{aligned}
$$

ここで $N_{s}$ と $N_r$ はそれぞれ共有エキスパートとルーティングされるエキスパートの数、$\mathop{\mathrm{FFN}}^{(s)}_{i}(\cdot)$ と $\mathop{\mathrm{FFN}}^{(r)}_{i}(\cdot)$ はそれぞれ $i$ 番目の共有エキスパートと $i$ 番目のルーティングされるエキスパート、$K_{r}$ は活性化されるルーティングエキスパートの数、$g_{i,t}$ は $i$ 番目のエキスパートのゲート値、$s_{i,t}$ はトークンとエキスパートの親和度、$\mathbf{e}_{i}$ はこの層における $i$ 番目のルーティングされるエキスパートの重心、$\mathop{\mathrm{Topk}}(\cdot, K)$ は $t$ 番目のトークンとすべてのルーティングされるエキスパートについて計算された親和度スコアのうち上位 $K$ 個からなる集合を表します。

<span id="section-2-2-2"></span>

#### 2.2.2 デバイス制限付きルーティング

MoE 関連の通信コストを制限するため、デバイス制限付きルーティング機構を設計しました。エキスパート並列を採用すると、ルーティングされるエキスパートは複数のデバイスに分散されます。各トークンについて、その MoE 関連の通信頻度はターゲットエキスパートがカバーするデバイス数に比例します。DeepSeekMoE の細粒度なエキスパート分割により活性化されるエキスパート数が多くなる可能性があるため、エキスパート並列を適用すると MoE 関連の通信はより高コストになります。

DeepSeek-V2 では、ルーティングされるエキスパートの単純な top-K 選択に加えて、各トークンのターゲットエキスパートが最大 $M$ 個のデバイスに分散されることも保証します。具体的には、各トークンについて、まず親和度スコアが最も高いエキスパートを持つ $M$ 個のデバイスを選択します。次に、これら $M$ 個のデバイス上のエキスパート間で top-K 選択を行います。実際には、$M \geq 3$ のとき、デバイス制限付きルーティングは制限のない top-K ルーティングとほぼ一致する良好な性能を達成できることを確認しています。

<span id="section-2-2-3"></span>

#### 2.2.3 負荷バランスのための補助損失

自動学習されるルーティング戦略について、負荷バランスを考慮します。まず、不均衡な負荷はルーティング崩壊 [Sha17] のリスクを高め、一部のエキスパートが十分に訓練・利用されないことを防ぎます。次に、エキスパート並列を採用する場合、不均衡な負荷は計算効率を低下させます。DeepSeek-V2 の学習では、エキスパートレベルの負荷バランス（$\mathcal{L}_{\mathrm{ExpBal}}$）、デバイスレベルの負荷バランス（$\mathcal{L}_{\mathrm{DevBal}}$）、通信バランス（$\mathcal{L}_{\mathrm{CommBal}}$）をそれぞれ制御する三種類の補助損失を設計しました。

**エキスパートレベルのバランス損失。** ルーティング崩壊のリスクを軽減するため、エキスパートレベルのバランス損失 [Fed22, Lep20] を用います。

$$
\begin{aligned}
    \mathcal{L}_{\mathrm{ExpBal}} & = \alpha_1 \sum_{i=1}^{N_r}{f_i P_i}, \\
    f_i & = \frac{N_r}{K_r T} \sum_{t=1}^{T}{ \mathbb{1}( \mathrm{Token}\ t\ \text{がエキスパート}\ i\ \text{を選択} )}, \\
    P_i & = \frac{1}{T} \sum_{t=1}^{T}{s_{i,t}},
\end{aligned}
$$

ここで $\alpha_1$ はエキスパートレベルのバランス係数と呼ばれるハイパーパラメータ、$\mathbb{1}(\cdot)$ は指示関数、$T$ はシーケンス内のトークン数を表します。

**デバイスレベルのバランス損失。** エキスパートレベルのバランス損失に加えて、異なるデバイス間での計算のバランスを確保するため、デバイスレベルのバランス損失も設計しました。DeepSeek-V2 の学習過程では、すべてのルーティングされるエキスパートを $D$ 個のグループ $\{\mathcal{E}_1, \mathcal{E}_2, ..., \mathcal{E}_D \}$ に分割し、各グループを単一のデバイスに配置します。デバイスレベルのバランス損失は次のように計算されます。

$$
\begin{aligned}
    \mathcal{L}_{\mathrm{DevBal}} & = \alpha_{2} \sum_{i=1}^{D}{f_i^{\prime} P_i^{\prime}}, \\
    f_i^{\prime} & = \frac{1}{|\mathcal{E}_i|} \sum_{j \in \mathcal{E}_i}{ f_j }, \\
    P_i^{\prime} & = \sum_{j \in \mathcal{E}_i}{ P_j },
\end{aligned}
$$

ここで $\alpha_{2}$ はデバイスレベルのバランス係数と呼ばれるハイパーパラメータです。

**通信バランス損失。** 最後に、各デバイスの通信がバランスされることを保証する通信バランス損失を導入します。デバイス制限付きルーティング機構は各デバイスの送信通信が有界であることを保証しますが、あるデバイスが他のデバイスより多くのトークンを受け取る場合、実際の通信効率も影響を受けます。この問題を軽減するため、次のような通信バランス損失を設計しました。

$$
\begin{aligned}
    \mathcal{L}_{\mathrm{CommBal}} & = \alpha_{3} \sum_{i=1}^{D}{f_i^{\prime\prime} P_i^{\prime\prime}}, \\
    f_i^{\prime\prime} & = \frac{D}{M T} \sum_{t=1}^{T}{ \mathbb{1}( \mathrm{Token}\ t\ \text{がデバイス}\ i\ \text{へ送信} )}, \\
    P_i^{\prime\prime} & = \sum_{j \in \mathcal{E}_i}{ P_j },
\end{aligned}
$$

ここで $\alpha_{3}$ は通信バランス係数と呼ばれるハイパーパラメータです。デバイス制限付きルーティング機構は、各デバイスが最大 $M T$ 個の隠れ状態を他のデバイスへ送信することを保証する原理で動作します。同時に、通信バランス損失は各デバイスが他のデバイスから約 $M T$ 個の隠れ状態を受け取ることを促すために用いられます。通信バランス損失はデバイス間の情報の均衡した交換を保証し、効率的な通信を促進します。

<span id="section-2-2-4"></span>

#### 2.2.4 トークン切り捨て戦略

バランス損失は均衡の取れた負荷を促すことを目的としていますが、厳密な負荷バランスを保証できないことを認識することが重要です。不均衡な負荷による計算の浪費をさらに軽減するため、学習中にデバイスレベルのトークン切り捨て戦略を導入します。この手法はまず各デバイスの平均計算予算を計算します。これは各デバイスの容量係数が 1.0 に相当することを意味します。次に、[Riq21] に触発され、各デバイス上で親和度スコアが最も低いトークンを計算予算に達するまで切り捨てます。さらに、約 10% の学習シーケンスに属するトークンは決して切り捨てられないことを保証します。これにより、効率要件に応じて推論中にトークンを切り捨てるかどうかを柔軟に決定でき、学習と推論の間の一貫性を常に確保できます。

<span id="section-3"></span>

## 3 事前学習

<span id="section-3-1"></span>

### 3.1 実験設定

<span id="section-3-1-1"></span>

#### 3.1.1 データ構築

DeepSeek 67B [Dee24e] と同じデータ処理段階を維持しつつ、データ量を拡充しデータ品質を高めました。事前学習コーパスを拡大するため、インターネットデータの可能性を探り、クリーニングプロセスを最適化することで、誤って削除された大量のデータを回復しました。さらに、中国語インターネット上で利用可能なコーパスをより活用するため、中国語データをより多く取り入れました。データ量に加えてデータ品質にも焦点を当てています。さまざまなソースからの高品質データで事前学習コーパスを豊かにし、同時に品質ベースのフィルタリングアルゴリズムを改善しました。改善されたアルゴリズムは、大量の有益でないデータを除去しつつ、価値のあるデータをほぼ保持することを保証します。さらに、特定の地域文化から導入されるデータバイアスを軽減するため、事前学習コーパスから論争の的となる内容をフィルタリングします。このフィルタリング戦略の影響に関する詳細な議論は [第 10 節](#section-10) に示します。

DeepSeek 67B と同じトークナイザを採用します。これは Byte-level Byte-Pair Encoding（BBPE）アルゴリズムに基づいて構築され、語彙サイズは 100K です。トークナイズされた事前学習コーパスは 8.1T トークンを含み、中国語トークンは英語よりも約 12% 多くなっています。

<span id="section-3-1-2"></span>

#### 3.1.2 ハイパーパラメータ

**モデルのハイパーパラメータ。** Transformer 層数を 60、隠れ次元を 5120 に設定します。すべての学習可能パラメータは標準偏差 0.006 でランダム初期化します。MLA では、アテンションヘッド数 $n_h$ を 128、ヘッドごとの次元 $d_h$ を 128 に設定します。KV 圧縮次元 $d_c$ は 512、クエリ圧縮次元 $d_c^{\prime}$ は 1536 に設定します。分離されたクエリとキーについては、ヘッドごとの次元 $d_h^R$ を 64 に設定します。[Dai24] に従い、最初の層を除くすべての FFN を MoE 層に置き換えます。各 MoE 層は 2 個の共有エキスパートと 160 個のルーティングされるエキスパートからなり、各エキスパートの中間隠れ次元は 1536 です。ルーティングされるエキスパートのうち、各トークンにつき 6 個のエキスパートが活性化されます。さらに、低ランク圧縮と細粒度のエキスパート分割は層の出力スケールに影響します。したがって、実際には圧縮された潜在ベクトルの後に追加の RMS Norm 層を配置し、幅ボトルネック（すなわち圧縮された潜在ベクトルとルーティングされるエキスパートの中間隠れ状態）で追加のスケーリング係数を乗じて、安定した学習を保証します。この構成のもと、DeepSeek-V2 は総パラメータ 236B を含み、各トークンにつき 21B を活性化します。

**学習のハイパーパラメータ。** AdamW オプティマイザ [Los17] を用い、ハイパーパラメータは $\beta_1=0.9$、$\beta_2=0.95$、$\mathrm{weight\_decay}=0.1$ に設定します。学習率はウォームアップとステップ減衰戦略 [Dee24e] でスケジュールします。最初に、最初の 2K ステップで学習率は 0 から最大値へ線形に増加します。その後、約 60% のトークンを学習した後に学習率を 0.316 倍し、約 90% のトークンを学習した後に再度 0.316 倍します。最大学習率は $2.4 \times 10^{-4}$ に設定し、勾配クリッピングノルムは 1.0 に設定します。バッチサイズスケジューリング戦略も用い、最初の 225B トークンの学習でバッチサイズを 2304 から 9216 へ段階的に増やし、残りの学習では 9216 を維持します。最大シーケンス長を 4K に設定し、8.1T トークンで DeepSeek-V2 を学習します。パイプライン並列を利用してモデルの異なる層を異なるデバイスへ配置し、各層についてルーティングされるエキスパートは 8 台のデバイスに均等に配置されます（$D=8$）。デバイス制限付きルーティングについては、各トークンは最大 3 台のデバイスへ送信されます（$M=3$）。バランス損失については、$\alpha_{1}$ を 0.003、$\alpha_{2}$ を 0.05、$\alpha_{3}$ を 0.02 に設定します。加速のため学習中にトークン切り捨て戦略を採用しますが、評価ではトークンを切り捨てません。

<span id="section-3-1-3"></span>

#### 3.1.3 インフラストラクチャ

DeepSeek-V2 は、当社のエンジニアが社内で開発した効率的で軽量な学習フレームワークである HAI-LLM フレームワーク [Hig23] に基づいて学習されます。16 方向のゼロバブルパイプライン並列 [Qi23]、8 方向のエキスパート並列 [Lep20]、ZeRO-1 データ並列 [Raj20] を採用しています。DeepSeek-V2 は活性化パラメータが比較的少なく、活性化メモリを節約するために一部の演算子が再計算されることを踏まえ、テンソル並列なしで学習でき、それによって通信オーバーヘッドが減少します。さらに学習効率を高めるため、共有エキスパートの計算をエキスパート並列の all-to-all 通信とオーバーラップさせます。また、通信、ルーティングアルゴリズム、異なるエキスパート間の融合線形計算のために、より高速な CUDA カーネルをカスタマイズしました。さらに MLA は、FlashAttention-2 [Dao24] の改良版に基づいて最適化されています。

すべての実験は NVIDIA H800 GPU を搭載したクラスタで実施しました。H800 クラスタの各ノードは 8 個の GPU を含み、ノード内では NVLink と NVSwitch で接続されます。ノード間には InfiniBand 相互接続を用いて通信を促進します。

<span id="figure-04"></span>

![DeepSeek-V2 図 4](../../papers/deepseek-v2/figure-04.png)

**図 4.** 「Needle In A Haystack」（NIAH）テストの評価結果。DeepSeek-V2 は最大 128K までのすべてのコンテキストウィンドウ長で良好な性能を示します。

<span id="section-3-1-4"></span>

#### 3.1.4 長いコンテキストの拡張

DeepSeek-V2 の初期事前学習後、YaRN [Pen23] を用いて既定のコンテキストウィンドウ長を 4K から 128K へ拡張します。YaRN は、RoPE [Su24] を運ぶ役割を担う分離された共有キー $\mathbf{k}^R_t$ に特別に適用されました。YaRN については、スケール $s$ を 40、$\alpha$ を 1、$\beta$ を 32、目標最大コンテキスト長を 160K に設定します。これらの設定の下で、モデルが 128K のコンテキスト長に良好に応答することが期待できます。独自のアテンション機構のため、元の YaRN とはわずかに異なり、アテンションエントロピーを調整するために長さスケーリング係数を調整します。係数 $\sqrt{t}$ は $\sqrt{t} = 0.0707 \ln{s} + 1$ と計算され、パープレキシティを最小化することを目指します。

さらに、シーケンス長 32K、バッチサイズ 576 シーケンスで 1000 ステップ追加学習します。学習は 32K のシーケンス長のみで行われますが、128K のコンテキスト長で評価したとき、モデルは依然として頑健な性能を示します。[図 4](#figure-04) に示すように、「Needle In A Haystack」（NIAH）テストの結果は、DeepSeek-V2 が最大 128K までのすべてのコンテキストウィンドウ長で良好な性能を示すことを示しています。

<span id="section-3-2"></span>

### 3.2 評価

<span id="section-3-2-1"></span>

#### 3.2.1 評価ベンチマーク

DeepSeek-V2 はバイリンガルコーパスで事前学習されているため、英語と中国語の一連のベンチマークで評価します。評価は HAI-LLM フレームワークに統合された社内評価フレームワークに基づきます。含まれるベンチマークは以下のように分類して列挙されます。下線付きのベンチマークは中国語です。

**多分野の多肢選択** データセットには MMLU [Hen20]、C-Eval [Hua23]、CMMLU [Li23e] が含まれます。

**言語理解と推論** データセットには HellaSwag [Zel19]、PIQA [Bis20]、ARC [Cla18]、BigBench Hard（BBH）[Suz22] が含まれます。

**閉じた本の質問応答** データセットには TriviaQA [Jos17] と NaturalQuestions [Kwi19a] が含まれます。

**読解** データセットには RACE [Lai17]、DROP [Dua19]、C3 [Sun19c]、CMRC [Cui19] が含まれます。

**参照曖昧性解消** データセットには WinoGrande [Sak19] と CLUEWSC [Xu20] が含まれます。

**言語モデリング** データセットには Pile [Gao20] が含まれます。

**中国語理解と文化** データセットには CHID [Zhe19] と CCPM [Li21e] が含まれます。

**数学** データセットには GSM8K [Cob21]、MATH [Hen21]、CMath [Wei23b] が含まれます。

**コード** データセットには HumanEval [Che21e]、MBPP [Aus21b]、CRUXEval [Gu24] が含まれます。

**標準化試験** には AGIEval [Zho23] が含まれます。AGIEval は英語と中国語の両方のサブセットを含むことに注意してください。

以前の研究 [Dee24e] に従い、HellaSwag、PIQA、WinoGrande、RACE-Middle、RACE-High、MMLU、ARC-Easy、ARC-Challenge、CHID、C-Eval、CMMLU、C3、CCPM を含むデータセットにはパープレキシティに基づく評価を、TriviaQA、NaturalQuestions、DROP、MATH、GSM8K、HumanEval、MBPP、CRUXEval、BBH、AGIEval、CLUEWSC、CMRC、CMath には生成ベースの評価を採用します。さらに、Pile-test には言語モデリングベースの評価を行い、異なるトークナイザを持つモデル間の公正な比較を保証するため Bits-Per-Byte（BPB）を指標として使用します。

これらのベンチマークの直感的な概観のため、各ベンチマークの評価形式も [第 12 節](#section-12) に示します。

<span id="section-3-2-2"></span>

#### 3.2.2 評価結果

<span id="table-02"></span>

![DeepSeek-V2 表 2](../../papers/deepseek-v2/table-02.png)

**表 2.** DeepSeek-V2 と他の代表的なオープンソースモデルの比較。すべてのモデルを社内フレームワークで評価し、同じ評価設定を共有します。**太字** は最良を、下線は次善を表します。0.3 未満の差のスコアは同一水準とみなします。わずか 21B の活性化パラメータで、DeepSeek-V2 はオープンソースモデルの中でトップクラスの性能を達成します。

[表 2](#table-02) では、DeepSeek-V2 を、DeepSeek 67B [Dee24e]（以前のリリース）、Qwen1.5 72B [Bai23b]、LLaMA3 70B [Dub24]、Mixtral 8x22B [Mis24] を含むいくつかの代表的なオープンソースモデルと比較します。これらすべてのモデルを社内評価フレームワークで評価し、同じ評価設定を共有することを保証します。全体として、わずか 21B の活性化パラメータで、DeepSeek-V2 はほぼすべてのベンチマークで DeepSeek 67B を大幅に上回り、オープンソースモデルの中でトップクラスの性能を達成します。

さらに、DeepSeek-V2 をオープンソースの競合と一つずつ丁寧に比較します。

1. 中国語と英語の両方をサポートする別のモデルである Qwen1.5 72B と比べ、DeepSeek-V2 はほとんどの英語、コード、数学ベンチマークで圧倒的な優位性を示します。中国語ベンチマークについては、Qwen1.5 72B は多分野の多肢選択タスクでより良い性能を示す一方、DeepSeek-V2 は他のタスクで同等かそれ以上です。CHID ベンチマークについては、Qwen1.5 72B のトークナイザが当社の評価フレームワークでエラーを起こすため、Qwen1.5 72B の CHID スコアは空欄にしています。
2. Mixtral 8x22B と比べ、DeepSeek-V2 は英語の常識的知識に密接に関連する TriviaQA、NaturalQuestions、HellaSwag を除き、同等かそれ以上の英語性能を達成します。特筆すべきは、DeepSeek-V2 が MMLU で Mixtral 8x22B を上回ることです。コードと数学のベンチマークでは、DeepSeek-V2 は Mixtral 8x22B と同等の性能を示します。Mixtral 8x22B は中国語データで特別に学習されていないため、その中国語能力は DeepSeek-V2 に大きく遅れを取ります。
3. LLaMA3 70B と比べ、DeepSeek-V2 は英語トークンの 4 分の 1 未満で学習されています。したがって、DeepSeek-V2 は基本的な英語能力において LLaMA3 70B と依然としてわずかな差があることを認めます。しかし、はるかに少ない学習トークンと活性化パラメータでも、DeepSeek-V2 は LLaMA3 70B と同等のコードおよび数学能力を示します。また、バイリンガル言語モデルとして、DeepSeek-V2 は中国語ベンチマークで LLaMA3 70B を圧倒的に上回ります。

最後に、特定の先行研究 [Hu24] が事前学習段階に SFT データを組み込んでいる一方、DeepSeek-V2 は事前学習中に SFT データに一度も触れていないことに言及する価値があります。

<span id="section-3-2-3"></span>

#### 3.2.3 学習と推論の効率

**学習コスト。** DeepSeek-V2 は各トークンにつき活性化するパラメータが少なく、DeepSeek 67B より少ない FLOPs を必要とするため、理論的には DeepSeek 67B を学習するより DeepSeek-V2 を学習する方が経済的です。MoE モデルの学習は追加の通信オーバーヘッドを導入しますが、オペレータと通信の最適化により、DeepSeek-V2 の学習は比較的高いモデル FLOPs 利用率（MFU）を達成できます。H800 クラスタでの実際の学習では、1 兆トークンあたりの学習につき、DeepSeek 67B は 300.6K GPU 時間を要する一方、DeepSeek-V2 はわずか 172.8K GPU 時間しか必要としません。すなわち、スパースな DeepSeek-V2 は密な DeepSeek 67B と比べて学習コストを 42.5% 節約できます。

**推論効率。** DeepSeek-V2 をサービス展開するため、まずそのパラメータを FP8 精度へ変換します。さらに、DeepSeek-V2 に KV キャッシュ量子化 [Hoo24, Zha24e] を施し、その KV キャッシュ内の各要素を平均 6 ビットにさらに圧縮します。MLA とこれらの最適化により、実際に展開された DeepSeek-V2 は DeepSeek 67B よりはるかに少ない KV キャッシュを必要とし、そのためより大きなバッチサイズを提供できます。実際に展開された DeepSeek 67B サービスのプロンプト長と生成長の分布に基づいて、DeepSeek-V2 の生成スループットを評価します。8 個の H800 GPU を搭載した単一ノードで、DeepSeek-V2 は毎秒 5 万トークンを超える生成スループットを達成し、これは DeepSeek 67B の最大生成スループットの 5.76 倍です。さらに、DeepSeek-V2 のプロンプト入力スループットは毎秒 10 万トークンを超えます。

<span id="section-4"></span>

## 4 アライメント

<span id="section-4-1"></span>

### 4.1 教師ありファインチューニング

以前の研究 [Dee24e] を土台に、150 万インスタンスからなる指示チューニングデータセットを厳選しました。有用性のための 120 万インスタンスと安全性のための 30 万インスタンスを含みます。初期バージョンと比べ、幻覚的な応答を軽減し執筆能力を高めるため、データ品質を改善しました。DeepSeek-V2 を 2 エポックでファインチューニングし、学習率は $5 \times 10^{-6}$ に設定します。DeepSeek-V2 Chat（SFT）の評価には、いくつかの代表的な多肢選択タスク（MMLU と ARC）を除き、主に生成ベースのベンチマークを含めます。また、DeepSeek-V2 Chat（SFT）に対して指示追従評価（IFEval）[Zho23a] を行い、プロンプトレベルの緩い正解率を指標とします。さらに、2023 年 9 月 1 日から 2024 年 4 月 1 日までの LiveCodeBench [Jai25a] 問題を用いてチャットモデルを評価します。標準ベンチマークに加えて、MT-Bench [Sto23e]、AlpacaEval 2.0 [Dub24a]、AlignBench [Liu23m] を含むオープンエンド対話ベンチマークでさらに評価します。比較のため、当社の評価フレームワークと設定で Qwen1.5 72B Chat、LLaMA-3-70B Instruct、Mistral-8x22B Instruct も評価します。DeepSeek 67B Chat については、以前のリリースで報告された評価結果を直接参照します。

<span id="section-4-2"></span>

### 4.2 強化学習

DeepSeek-V2 の潜在能力をさらに引き出し、人間の好みに合わせるため、強化学習（RL）を実施してその好みを調整します。

**強化学習アルゴリズム。** RL の学習コストを節約するため、通常ポリシーモデルと同じ規模の批評家モデルを捨て、グループスコアからベースラインを推定する Group Relative Policy Optimization（GRPO）[Sha24d] を採用します。具体的には、各質問 $q$ について、GRPO は古いポリシー $\pi_{\theta_{\mathrm{old}}}$ から出力のグループ $\{o_1, o_2, \cdots, o_G\}$ をサンプリングし、以下の目的を最大化することでポリシーモデル $\pi_{\theta}$ を最適化します。

<span id="equation-25"></span>

$$
\begin{aligned}
    \mathcal{J}_{\mathrm{GRPO}}(\theta) &= \mathbb{E}{[q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{\mathrm{old}}}(O|q)]}  \\
    & \frac{1}{G}\sum_{i=1}^G \left( \min \left( \frac{\pi_\theta(o_i |q)}{\pi_{\theta_{\mathrm{old}}}(o_i |q)} A_i, \mathop{\mathrm{clip}} \left( \frac{\pi_\theta(o_i |q)}{\pi_{\theta_{\mathrm{old}}}(o_i |q)}, 1 - \epsilon, 1 + \epsilon \right)  A_i \right) - \beta \mathbb{D}_{\mathrm{KL}}\left(\pi_{\theta} \| \pi_{\mathrm{ref}}\right)\right) ,
\end{aligned}
$$

$$
\begin{aligned}
    \mathbb{D}_{\mathrm{KL}}\left(\pi_{\theta} \| \pi_{\mathrm{ref}}\right) = \frac{\pi_{\mathrm{ref}}(o_i|q)}{\pi_{\theta}(o_i|q)}- \log\frac{\pi_{\mathrm{ref}}(o_i|q)}{\pi_{\theta}(o_i|q)} - 1,
\end{aligned}
$$

ここで $\epsilon$ と $\beta$ はハイパーパラメータ、$A_i$ は各グループ内の出力に対応する一連の報酬 $\{r_1, r_2, \ldots, r_G\}$ を用いて計算されるアドバンテージです。

$$
\begin{aligned}
    A_i = \frac{r_i - {\mathrm{mean}(\{r_1, r_2, \cdots, r_G\})}}{{\mathrm{std}(\{r_1, r_2, \cdots, r_G\})}}.
\end{aligned}
$$

**学習戦略。** 予備実験では、コードや数学のプロンプトなどの推論データに対する RL 学習が、一般的なデータに対する学習とは異なる独自の特性を示すことが分かりました。例えば、モデルの数学的・コーディング能力は、より長い学習ステップにわたって改善し続けることができます。そこで、まず推論アライメントを行い、その後人間の好みアライメントを行う二段階の RL 学習戦略を採用します。第一段階の推論アライメントでは、コードと数学の推論タスクのための報酬モデル $\mathit{RM}_{\mathrm{reasoning}}$ を学習し、$\mathit{RM}_{\mathrm{reasoning}}$ のフィードバックでポリシーモデルを最適化します。

$$
\begin{aligned}
    r_i=\mathit{RM}_{\mathrm{reasoning}}(o_i).
\end{aligned}
$$

第二段階の人間の好みアライメントでは、有用性報酬モデル $\mathit{RM}_{\mathrm{helpful}}$、安全性報酬モデル $\mathit{RM}_{\mathrm{safety}}$、ルールベース報酬モデル $\mathit{RM}_{\mathrm{rule}}$ から報酬を得るマルチ報酬フレームワークを採用します。応答 $o_i$ の最終報酬は

$$
\begin{aligned}
    r_i = c_1 \cdot \mathit{RM}_{\mathrm{helpful}}(o_i) + c_2 \cdot \mathit{RM}_{\mathrm{safety}}(o_i) + c_3 \cdot \mathit{RM}_{\mathrm{rule}}(o_i),
\end{aligned}
$$

ここで $c_1$、$c_2$、$c_3$ は対応する係数です。

RL 学習で重要な役割を果たす信頼できる報酬モデルを得るため、選好データを慎重に収集し、品質フィルタリングと比率調整を綿密に行います。コンパイラフィードバックに基づいてコード選好データを、正解ラベルに基づいて数学選好データを得ます。報酬モデルの学習では、DeepSeek-V2 Chat（SFT）で報酬モデルを初期化し、ポイントワイズまたはペアワイズ損失で学習します。実験では、RL 学習がモデルの潜在能力を十分に引き出して活性化し、可能な応答の中から正しく満足のいく答えを選択できるようにすることが観察されました。

**学習効率のための最適化。** 超大規模モデルでの RL 学習は学習フレームワークに高い要求を課します。GPU メモリと RAM の圧力を管理しつつ、学習速度を維持するための慎重なエンジニアリング最適化が必要です。この目標のため、以下のエンジニアリング最適化を実装しました。

1. まず、学習と推論にそれぞれ異なる並列戦略を採用し、より高い GPU 利用率を達成するハイブリッドエンジンを提案します。
2. 次に、大規模バッチサイズの vLLM [Kwo23] を推論バックエンドとして利用し、推論速度を加速します。
3. 第三に、モデルを CPU へオフロードし GPU へ再読み込みするスケジューリング戦略を慎重に設計し、学習速度とメモリ消費の間でほぼ最適なバランスを達成します。

<span id="section-4-3"></span>

### 4.3 評価結果

**標準ベンチマークでの評価。** まず、DeepSeek-V2 Chat（SFT）と DeepSeek-V2 Chat（RL）を標準ベンチマークで評価します。特筆すべきは、DeepSeek-V2 Chat（SFT）がそのベース版と比べ、GSM8K、MATH、HumanEval の評価で大幅な改善を示すことです。この進歩は、数学とコードに関連する相当量の内容を含む SFT データを取り入れたことに起因します。さらに、DeepSeek-V2 Chat（RL）は数学とコードのベンチマークで性能をさらに高めます。より多くのコードと数学の評価を [第 11 節](#section-11) に示します。

他のモデルとの比較については、まず DeepSeek-V2 Chat（SFT）を Qwen1.5 72B Chat と比較し、DeepSeek-V2 Chat（SFT）がほぼすべての英語、数学、コードベンチマークで Qwen1.5 72B Chat を上回ることを見いだしました。中国語ベンチマークでは、DeepSeek-V2 Chat（SFT）は多分野の多肢選択タスクで Qwen1.5 72B Chat よりわずかに低いスコアを示し、これはベース版で観察された性能と一致します。最先端のオープンソース MoE モデルである Mixtral 8x22B Instruct と比較すると、DeepSeek-V2 Chat（SFT）は NaturalQuestions と IFEval を除くほとんどのベンチマークでより良い性能を示します。さらに、最先端のオープンソースモデルである LLaMA3 70B Chat と比較すると、DeepSeek-V2 Chat（SFT）はコードと数学関連のベンチマークで同様の性能を示します。LLaMA3 70B Chat は MMLU と IFEval でより良い性能を示す一方、DeepSeek-V2 Chat（SFT）は中国語タスクでより強い性能を見せます。最終的に、DeepSeek-V2 Chat（RL）は DeepSeek-V2 Chat（SFT）と比べ、数学とコードの両方のタスクでさらに強化された性能を示します。これらの比較は、さまざまな領域と言語における他の言語モデルに対する DeepSeek-V2 Chat の強みを浮き彫りにします。

<span id="table-03"></span>

![DeepSeek-V2 表 3](../../papers/deepseek-v2/table-03.png)

**表 3.** DeepSeek-V2 Chat（SFT）、DeepSeek-V2 Chat（RL）と他の代表的なオープンソースチャットモデルの比較。TriviaQA と NaturalQuestions については、LLaMA3 70B Instruct のようなチャットモデルが、少数ショット設定で通常指定される形式制約を厳密には守らない場合があることに注意する価値があります。その結果、当社の評価フレームワークでは特定のモデルを過小評価する可能性があります。

**オープンエンド生成での評価。** オープンエンド対話ベンチマークでモデルの追加評価を進めます。英語のオープンエンド対話生成には、MT-Bench と AlpacaEval 2.0 をベンチマークとして用います。[表 4](#table-04) に示す評価結果は、DeepSeek-V2 Chat（RL）が DeepSeek-V2 Chat（SFT）に対して顕著な性能優位性を持つことを示しています。この結果は、より良いアライメントを達成する上での RL 学習の有効性を示しています。他のオープンソースモデルと比べ、DeepSeek-V2 Chat（RL）は両ベンチマークで Mistral 8x22B Instruct と Qwen1.5 72B Chat より優れた性能を示します。LLaMA3 70B Instruct と比較すると、DeepSeek-V2 Chat（RL）は MT-Bench で競争力のある性能を示し、AlpacaEval 2.0 では顕著に上回ります。これらの結果は、特に指示ベースの対話タスクにおいて、高品質で文脈に関連した応答を生成する DeepSeek-V2 Chat（RL）の強力な性能を強調します。

<span id="table-04"></span>

![DeepSeek-V2 表 4](../../papers/deepseek-v2/table-04.png)

**表 4.** 英語のオープンエンド対話評価。AlpacaEval 2.0 では長さ制御勝率を指標として用います。

さらに、AlignBench に基づいて中国語のオープンエンド生成能力を評価します。[表 5](#table-05) に示すように、DeepSeek-V2 Chat（RL）は DeepSeek-V2 Chat（SFT）に対してわずかな優位性を示します。特筆すべきは、DeepSeek-V2 Chat（SFT）がすべてのオープンソース中国語モデルを大幅に上回ることです。中国語の推論と言語の両方で、次善のオープンソースモデルである Qwen1.5 72B Chat を顕著に上回ります。さらに、DeepSeek-V2 Chat（SFT）と DeepSeek-V2 Chat（RL）はともに GPT-4-0613 と ERNIEBot 4.0 を上回り、中国語をサポートするトップクラスの LLM における当社モデルの地位を固めます。具体的には、DeepSeek-V2 Chat（RL）は中国語の言語理解で顕著な性能を示し、GPT-4-Turbo-1106-Preview を含むすべてのモデルを上回ります。一方、DeepSeek-V2 Chat（RL）の推論能力は Erniebot-4.0 や GPT-4 のような巨大モデルには依然として及ばない点があります。

<span id="table-05"></span>

![DeepSeek-V2 表 5](../../papers/deepseek-v2/table-05.png)

**表 5.** GPT-4-0613 が評価した AlignBench リーダーボード。モデルは総合スコアの降順で並べられています。* を付けたモデルは、元論文で報告された結果を参照するのではなく、API サービスまたはオープンウェイトモデルを通じて評価したことを表します。Erniebot-4.0 と Moonshot の接尾辞は、API を呼び出した際のタイムスタンプを表します。

<span id="section-4-4"></span>

### 4.4 議論

**SFT データ量。** 大規模な SFT コーパスの必要性をめぐる議論は、激しい論争の的となってきました。先行研究 [You24a, Zho24a] は、1 万未満の SFT データインスタンスで満足のいく結果が得られると主張しています。しかし、当社の実験では、1 万未満のインスタンスを用いると IFEval ベンチマークで顕著な性能低下を観察しました。考えられる説明は、言語モデルが特定の技能を身につけるために一定量のデータを必要とするというものです。必要なデータ量はモデル規模の増大に伴って減少するかもしれませんが、完全になくすことはできません。当社の観察は、LLM に望ましい能力を備えさせるために十分なデータが必要であるという重要な点を強調します。さらに、執筆やオープンエンドの質問を含むタスクでは、SFT データの品質も極めて重要です。

**強化学習のアライメント税。** 人間の好みアライメントの間、AI と人間の両方の評価者によるスコアの観点で、オープンエンド生成ベンチマークで顕著な性能向上を観察しました。しかし、「アライメント税」[Ouy22] という現象にも気づきました。すなわち、アライメントプロセスが BBH のような一部の標準ベンチマークの性能に悪影響を与える可能性があるということです。アライメント税を軽減するため、RL 段階でデータ処理と学習戦略の改善に多大な努力を払い、最終的に標準ベンチマークとオープンエンドベンチマークの性能の間に許容可能なトレードオフを達成しました。モデルの一般的な性能を損なわずに人間の好みへアライメントする方法を探求することは、将来の研究として価値ある方向です。

**オンライン強化学習。** 当社の選好アライメント実験では、オンライン手法がオフライン手法を大幅に上回ることが分かりました。そこで、DeepSeek-V2 をアライメントするためのオンライン RL フレームワークの実装に多大な努力を注ぎました。オンラインかオフラインかの選好アライメントの結論は文脈によって異なる可能性があり、それらをより徹底的に比較・分析することは将来の研究に残します。

<span id="section-5"></span>

## 5 結論、限界、今後の方向性

本稿では、128K のコンテキスト長をサポートする大型 MoE 言語モデル DeepSeek-V2 を紹介します。強力な性能に加え、MLA と DeepSeekMoE を含む革新的なアーキテクチャの恩恵により、経済的な学習と効率的な推論を特徴とします。実際には、DeepSeek 67B と比べ、DeepSeek-V2 は著しく強い性能を達成しつつ、学習コストを 42.5% 節約し、KV キャッシュを 93.3% 削減し、最大生成スループットを 5.76 倍に高めます。評価結果はさらに、わずか 21B の活性化パラメータで DeepSeek-V2 がオープンソースモデルの中でトップクラスの性能を達成し、最強のオープンソース MoE モデルになることを示しています。

DeepSeek-V2 とそのチャット版は、他の LLM に共通してみられる既知の限界を共有しています。事前学習後の継続的な知識更新の欠如、検証されていない助言のような非事実的情報を生成する可能性、幻覚を生じさせる可能性などです。さらに、当社のデータは主に中国語と英語で構成されるため、モデルは他の言語では習熟度が限られる可能性があります。中国語と英語以外のシナリオでは、注意して使用すべきです。

DeepSeek は、長期主義をもってオープンソースの大規模モデルへ投資し続け、人工汎用知能という目標に段階的に近づくことを目指します。

- 進行中の探求において、経済的な学習コストと推論コストを維持しながら MoE モデルをさらにスケールアップする手法を考案することに注力しています。次のステップの目標は、今後のリリースで GPT-4 と同等の性能を達成することです。
- 当社のアライメントチームは、世界中のユーザーにとって有用であるだけでなく、誠実で安全なモデルを開発することを目指し、モデルの強化に継続的に取り組んでいます。最終的な目標は、人間による監督の必要性を最小限に抑えつつ、モデルの価値を人間の価値に合わせることです。倫理的配慮と責任ある開発を優先することで、社会に積極的で有益な貢献を生み出すことに専念します。
- 現在、DeepSeek-V2 はテキストモダリティのみをサポートするよう設計されています。先を見据えた計画では、より広範なシナリオでその汎用性と有用性を高めるため、モデルが複数のモダリティをサポートできるようにする予定です。

<span id="section-6"></span>

## 6 貢献と謝辞

**研究・エンジニアリング。** Aixin Liu, Bingxuan Wang, Bo Liu, Chenggang Zhao, Chengqi Deng, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Erhang Li, Fangyun Lin, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Hanwei Xu, Hao Yang, Haowei Zhang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Qu, Jianzhong Guo, Jiashi Li, Jingyang Yuan, Junjie Qiu, Junxiao Song, Kai Dong, Kaige Gao, Kang Guan, Lean Wang, Lecong Zhang, Liang Zhao, Liyue Zhang, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Panpan Huang, Peiyi Wang, Qihao Zhu, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruizhe Pan, Runxin Xu, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shirong Ma, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Size Zheng, Tian Pei, Wangding Zeng, Wen Liu, Wenfeng Liang, Wenjun Gao, Wentao Zhang, Xiao Bi, Xiaohan Wang, Xiaodong Liu, Xiaokang Chen, Xiaotao Nie, Xin Liu, Xin Xie, Xingkai Yu, Xinyu Yang, Xuan Lu, Xuecheng Su, Y. Wu, Y.K. Li, Y.X. Wei, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yichao Zhang, Yiliang Xiong, Yilong Zhao, Ying He, Yishi Piao, Yixin Dong, Yixuan Tan, Yiyuan Liu, Yongji Wang, Yongqiang Guo, Yuduan Wang, Yuheng Zou, Yuxiang You, Yuxuan Liu, Z.Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhenda Xie, Zhewen Hao, Zhihong Shao, Zhuoshu Li, Zihan Wang, Zihui Gu, Zilin Li, Ziwei Xie

**データアノテーション。** Bei Feng, Hui Li, J.L. Cai, Jiaqi Ni, Lei Xu, Meng Li, Ning Tian, R.J. Chen, R.L. Jin, Ruyi Chen, S.S. Li, Shuang Zhou, Tian Yuan, Tianyu Sun, X.Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou, Y.X. Zhu, Yanhong Xu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Zhen Huang, Zhipeng Xu, Zhongyu Zhang

**事業・コンプライアンス。** Bin Wang, Dongjie Ji, Jian Liang, Jin Chen, Leyi Xia, Miaojun Wang, Mingming Li, Peng Zhang, Shaoqing Wu, Shengfeng Ye, T. Wang, W.L. Xiao, Wei An, Xianzu Wang, Ying Tang, Yukun Zha, Yuting Yan, Zhen Zhang, Zhiniu Wen

各役割内で、著者は名前のアルファベット順に記載されています。特に、Huazuo Gao と Wangding Zeng は MLA アーキテクチャの研究で重要な革新を行いました。さらに、位置埋め込みに関する有益な議論を提供してくれた Jianlin Su に感謝します。本稿に記載されていないものの DeepSeek-V2 に貢献したすべての人々に感謝します。DeepSeek は、AGI への道のりにおいて、革新、新規性、好奇心が不可欠であると信じています。

<span id="section-7"></span>

## 7 DeepSeek-V2-Lite: MLA と DeepSeekMoE を備えた 16B モデル

<span id="section-7-1"></span>

### 7.1 モデル説明

**アーキテクチャ。** DeepSeek-V2-Lite は 27 層、隠れ次元 2048 です。MLA も採用し、16 個のアテンションヘッドを持ち、各ヘッドの次元は 128 です。その KV 圧縮次元は 512 ですが、DeepSeek-V2 とはわずかに異なり、クエリは圧縮しません。分離されたクエリとキーについては、ヘッドごとの次元は 64 です。DeepSeek-V2-Lite も DeepSeekMoE を採用し、最初の層を除くすべての FFN が MoE 層に置き換えられています。各 MoE 層は 2 個の共有エキスパートと 64 個のルーティングされるエキスパートからなり、各エキスパートの中間隠れ次元は 1408 です。ルーティングされるエキスパートのうち、各トークンにつき 6 個が活性化されます。この構成の下、DeepSeek-V2-Lite は総パラメータ 15.7B を含み、各トークンにつき 2.4B を活性化します。

<span id="table-06"></span>

![DeepSeek-V2 表 6](../../papers/deepseek-v2/table-06.png)

**表 6.** DeepSeek-V2-Lite、DeepSeekMoE 16B、DeepSeek 7B の性能。

**学習の詳細。** DeepSeek-V2-Lite も DeepSeek-V2 と同じ事前学習コーパスでスクラッチから学習され、どの SFT データにも汚染されていません。AdamW オプティマイザを使用し、ハイパーパラメータは $\beta_1=0.9$、$\beta_2=0.95$、$\mathrm{weight\_decay}=0.1$ に設定します。学習率はウォームアップとステップ減衰戦略でスケジュールします。最初に、最初の 2K ステップで学習率は 0 から最大値へ線形に増加します。その後、約 80% のトークンを学習した後に学習率を 0.316 倍し、約 90% のトークンを学習した後に再度 0.316 倍します。最大学習率は $4.2 \times 10^{-4}$ に設定し、勾配クリッピングノルムは 1.0 に設定します。バッチサイズスケジューリング戦略は採用せず、一定のバッチサイズ 4608 シーケンスで学習します。事前学習では最大シーケンス長を 4K に設定し、5.7T トークンで DeepSeek-V2-Lite を学習します。パイプライン並列を利用して異なる層を異なるデバイスへ配置しますが、各層ではすべてのエキスパートが同じデバイスに配置されます。したがって、$\alpha_{1}=0.001$ の小さなエキスパートレベルのバランス損失のみを採用し、デバイスレベルのバランス損失と通信バランス損失は採用しません。事前学習後、DeepSeek-V2-Lite に対して長いコンテキスト拡張と SFT も行い、DeepSeek-V2-Lite Chat と呼ばれるチャットモデルを得ます。

<span id="table-07"></span>

![DeepSeek-V2 表 7](../../papers/deepseek-v2/table-07.png)

**表 7.** DeepSeek-V2-Lite Chat、DeepSeekMoE 16B Chat、DeepSeek 7B Chat の性能。

<span id="section-7-2"></span>

### 7.2 性能評価

**ベースモデル。** [表 6](#table-06) で DeepSeek-V2-Lite の性能を評価し、以前の小型ベースモデルと比較します。DeepSeek-V2-Lite は、特に推論、コーディング、数学で圧倒的な性能優位性を示します。

**チャットモデル。** [表 7](#table-07) で DeepSeek-V2-Lite Chat の性能を評価し、以前の小型チャットモデルと比較します。DeepSeek-V2-Lite も以前の小型チャットモデルを大きく上回ります。

<span id="section-8"></span>

## 8 MLA の完全な式

MLA の完全な計算過程を示すため、以下にその完全な式を提供します。

$$
\begin{aligned}
    \mathbf{c}_{t}^{Q} &= W^{\mathit{DQ}} \mathbf{h}_{t}, \\
    [\mathbf{q}_{t, 1}^{C};\mathbf{q}_{t, 2}^{C};...;\mathbf{q}_{t, n_{h}}^{C}] = \mathbf{q}_{t}^{C} &= W^{\mathit{UQ}} \mathbf{c}_{t}^{Q}, \\
    [\mathbf{q}_{t, 1}^{R};\mathbf{q}_{t, 2}^{R};...;\mathbf{q}_{t, n_{h}}^{R}] = \mathbf{q}_{t}^{R} &= \mathop{\mathrm{RoPE}}({W^{\mathit{QR}}} \mathbf{c}_{t}^{Q}), \\
    \mathbf{q}_{t, i} &= [\mathbf{q}_{t, i}^{C}; \mathbf{q}_{t, i}^{R}], \\
    \mathbf{c}_{t}^{\mathit{KV}} &= W^{\mathit{DKV}} \mathbf{h}_{t}, \\
    [\mathbf{k}_{t, 1}^{C};\mathbf{k}_{t, 2}^{C};...;\mathbf{k}_{t, n_{h}}^{C}] = \mathbf{k}_{t}^{C} &= W^{\mathit{UK}} \mathbf{c}_{t}^{\mathit{KV}}, \\
    \mathbf{k}_{t}^{R} &= \mathop{\mathrm{RoPE}}({W^{\mathit{KR}}} \mathbf{h}_{t}), \\
    \mathbf{k}_{t, i} &= [\mathbf{k}_{t, i}^{C}; \mathbf{k}_{t}^{R}], \\
    [\mathbf{v}_{t, 1}^{C};\mathbf{v}_{t, 2}^{C};...;\mathbf{v}_{t, n_{h}}^{C}] = \mathbf{v}_{t}^{C} &= W^{\mathit{UV}} \mathbf{c}_{t}^{\mathit{KV}}, \\
    \mathbf{o}_{t, i} &= \sum_{j=1}^{t} \mathop{\mathrm{Softmax}}_j(\frac{\mathbf{q}_{t, i}^\top \mathbf{k}_{j, i}}{\sqrt{d_{h} + d_{h}^{R}}}) \mathbf{v}_{j, i}^{C}, \\
    \mathbf{u}_{t} &= W^{O} [\mathbf{o}_{t, 1};\mathbf{o}_{t, 2};...;\mathbf{o}_{t, n_{h}}],
\end{aligned}
$$

ここで青い枠で囲まれたベクトルは生成時にキャッシュする必要があります。推論中、単純な式はアテンションのために $\mathbf{c}_{t}^{\mathit{KV}}$ から $\mathbf{k}_{t}^{C}$ と $\mathbf{v}_{t}^{C}$ を復元する必要があります。幸いにも、行列乗算の結合法則により、$W^{\mathit{UK}}$ を $W^{\mathit{UQ}}$ に、$W^{\mathit{UV}}$ を $W^{O}$ に吸収できます。したがって、各クエリについてキーと値を計算し出す必要はありません。この最適化により、推論中に $\mathbf{k}_{t}^{C}$ と $\mathbf{v}_{t}^{C}$ を再計算する計算オーバーヘッドを回避します。

<span id="section-9"></span>

## 9 アテンション機構のアブレーション

<span id="section-9-1"></span>

### 9.1 MHA、GQA、MQA のアブレーション

[表 8](#table-08) に、MHA、GQA、MQA をそれぞれ用いた 7B 密モデルの 4 つのハードベンチマークでの評価結果を示します。これら三つのモデルはすべて 1.33T トークンで学習され、アテンション機構以外は同じアーキテクチャを共有します。さらに、公平な比較のため、層数を調整してパラメータ数を約 7B に揃えます。表から、MHA がこれらのベンチマークで GQA と MQA に対して顕著な優位性を示すことが分かります。

<span id="table-08"></span>

![DeepSeek-V2 表 8](../../papers/deepseek-v2/table-08.png)

**表 8.** MHA、GQA、MQA をそれぞれ用いた 7B 密モデルの比較。MHA はハードベンチマークで GQA と MQA に対して顕著な優位性を示します。

<span id="section-9-2"></span>

### 9.2 MLA と MHA の比較

[表 9](#table-09) に、MLA と MHA をそれぞれ備えた MoE モデルの 4 つのハードベンチマークでの評価結果を示します。確かな結論のため、二つのスケールでモデルを学習・評価します。二つの小型 MoE モデルは合計約 16B パラメータを含み、1.33T トークンで学習します。二つの大型 MoE モデルは合計約 250B パラメータを含み、420B トークンで学習します。また、二つの小型 MoE モデルと二つの大型 MoE モデルは、それぞれアテンション機構以外は同じアーキテクチャを共有します。表から、MLA が MHA より良い性能を示すことが観察できます。より重要なのは、MLA が MHA より大幅に少ない KV キャッシュ（小型 MoE モデルで 14%、大型 MoE モデルで 4%）しか必要としないことです。

<span id="table-09"></span>

![DeepSeek-V2 表 9](../../papers/deepseek-v2/table-09.png)

**表 9.** ハードベンチマークでの MLA と MHA の比較。DeepSeek-V2 は MHA より良い性能を示しますが、必要な KV キャッシュは大幅に少ないです。

<span id="section-10"></span>

## 10 事前学習データのデバイアスに関する議論

事前学習データの準備中、地域文化の影響を受けた価値観など、論争の的となる内容を特定して除去し、これらの物議を醸す話題についてモデルが不要な主観的バイアスを示すのを避けました。その結果、特定の地域文化に密接に関連するテストセットでは、DeepSeek-V2 がわずかに劣ることを観察しました。例えば、MMLU で評価すると、DeepSeek-V2 は Mixtral 8x22B のような競合に対してほとんどのテストセットで同等かそれ以上の性能を達成しますが、主にアメリカ的価値観に関連する Humanity-Moral サブセットでは依然として遅れを取ります。

さらに、このサブセットについて手動分析を行いました。三人の教養ある人間のアノテータが、MMLU Humanity-Moral サブセットの 420 の道徳的シナリオについて独立にアノテーションを行いました。次に、それらのアノテーションと真値ラベルの間の一致度を計算しました。[表 10](#table-10) に示すように、三人の人間アノテータと真値ラベルは互いに低い一致度を示します。したがって、これらの価値に敏感なテストセットにおける DeepSeek-V2 の異常な性能を、事前学習コーパスのデバイアスへの取り組みに帰します。

<span id="table-10"></span>

![DeepSeek-V2 表 10](../../papers/deepseek-v2/table-10.png)

**表 10.** 三人の教養ある人間のアノテータが、MMLU Humanity-Moral サブセットの 420 の道徳的シナリオについて独立にアノテーションを行った。このサブセットでは DeepSeek-V2 と競合モデルが性能の一貫性を示さない。三人のアノテータと真値ラベルは互いに低い一致度を示す。これは Humanity-Moral サブセットの答えが特定の地域文化に応じて物議を醸す可能性があることを示します。

<span id="section-11"></span>

## 11 数学とコードに関する追加評価

評価は、数千の中国語の数学問題からなる SC-Math6 コーパスを用います。DeepSeek-V2 Chat（RL）は、オープンソースとクローズドソースの両方を含むすべての中国語 LLM を上回ります。

<span id="table-11"></span>

![DeepSeek-V2 表 11](../../papers/deepseek-v2/table-11.png)

**表 11.** SC-Math6 モデル推論レベル。"R Level" は Reasoning Level（推論レベル）、"Comp. Score" は Comprehensive Score（総合スコア）、"Reas. Steps Score" は Reasoning Steps Score（推論ステップスコア）、"OvrAcc Score" は Overall Accuracy Score（総合正解率スコア）を表します。

さらに、[図 5](#figure-05) で HumanEval と LiveCodeBench のさらなる結果を共有します。ここで LiveCodeBench の問題は 2023 年 9 月 1 日から 2024 年 4 月 1 日までの期間から選ばれています。図に示すように、DeepSeek-V2 Chat（RL）は LiveCodeBench でかなりの習熟度を示し、その Pass@1 スコアはいくつかの巨大モデルさえ上回ります。この性能は、ライブコーディングタスクに取り組む際の DeepSeek-V2 Chat（RL）の強力な能力を強調します。

<span id="figure-05"></span>

![DeepSeek-V2 図 5](../../papers/deepseek-v2/figure-05.png)

**図 5.** HumanEval と LiveCodeBench の評価結果。LiveCodeBench の問題は 2023 年 9 月 1 日から 2024 年 4 月 1 日までの期間から選ばれています。

<span id="section-12"></span>

## 12 評価形式

各ベンチマークの評価形式を [表 12](#table-12)-[表 37](#table-37) にそれぞれ示します。

<span id="table-12"></span>

![DeepSeek-V2 表 12](../../papers/deepseek-v2/table-12.png)

**表 12.** AGIEval の例。

<span id="table-13"></span>

![DeepSeek-V2 表 13](../../papers/deepseek-v2/table-13.png)

**表 13.** ARC の例。

<span id="table-14"></span>

![DeepSeek-V2 表 14](../../papers/deepseek-v2/table-14.png)

**表 14.** BBH の例。

<span id="table-15"></span>

![DeepSeek-V2 表 15](../../papers/deepseek-v2/table-15.png)

**表 15.** C-Eval の例。

<span id="table-16"></span>

![DeepSeek-V2 表 16](../../papers/deepseek-v2/table-16.png)

**表 16.** CHID の例。

<span id="table-17"></span>

![DeepSeek-V2 表 17](../../papers/deepseek-v2/table-17.png)

**表 17.** CLUEWSC の例。

<span id="table-18"></span>

![DeepSeek-V2 表 18](../../papers/deepseek-v2/table-18.png)

**表 18.** CMMLU の例。

<span id="table-19"></span>

![DeepSeek-V2 表 19](../../papers/deepseek-v2/table-19.png)

**表 19.** CMRC の例。

<span id="table-20"></span>

![DeepSeek-V2 表 20](../../papers/deepseek-v2/table-20.png)

**表 20.** CRUXEval-I の例。

<span id="table-21"></span>

![DeepSeek-V2 表 21](../../papers/deepseek-v2/table-21.png)

**表 21.** AGIEval 英語サブセットの例。

<span id="table-22"></span>

![DeepSeek-V2 表 22](../../papers/deepseek-v2/table-22.png)

**表 22.** CRUXEval-O の例。

<span id="table-23"></span>

![DeepSeek-V2 表 23](../../papers/deepseek-v2/table-23.png)

**表 23.** DROP の例。

<span id="table-24"></span>

![DeepSeek-V2 表 24](../../papers/deepseek-v2/table-24.png)

**表 24.** GSM8K の例。

<span id="table-25"></span>

![DeepSeek-V2 表 25](../../papers/deepseek-v2/table-25.png)

**表 25.** HumanEval の例。

<span id="table-26"></span>

![DeepSeek-V2 表 26](../../papers/deepseek-v2/table-26.png)

**表 26.** MATH の例。

<span id="table-27"></span>

![DeepSeek-V2 表 27](../../papers/deepseek-v2/table-27.png)

**表 27.** MBPP の例。

<span id="table-28"></span>

![DeepSeek-V2 表 28](../../papers/deepseek-v2/table-28.png)

**表 28.** MMLU の例。

<span id="table-29"></span>

![DeepSeek-V2 表 29](../../papers/deepseek-v2/table-29.png)

**表 29.** NaturalQuestions の例。

<span id="table-30"></span>

![DeepSeek-V2 表 30](../../papers/deepseek-v2/table-30.png)

**表 30.** PIQA の例。

<span id="table-31"></span>

![DeepSeek-V2 表 31](../../papers/deepseek-v2/table-31.png)

**表 31.** C3 の例。

<span id="table-32"></span>

![DeepSeek-V2 表 32](../../papers/deepseek-v2/table-32.png)

**表 32.** RACE の例。

<span id="table-33"></span>

![DeepSeek-V2 表 33](../../papers/deepseek-v2/table-33.png)

**表 33.** CMath の例。

<span id="table-34"></span>

![DeepSeek-V2 表 34](../../papers/deepseek-v2/table-34.png)

**表 34.** TriviaQA の例。

<span id="table-35"></span>

![DeepSeek-V2 表 35](../../papers/deepseek-v2/table-35.png)

**表 35.** CCPM の例。

<span id="table-36"></span>

![DeepSeek-V2 表 36](../../papers/deepseek-v2/table-36.png)

**表 36.** AGIEval 英語サブセットの例。

<span id="table-37"></span>

![DeepSeek-V2 表 37](../../papers/deepseek-v2/table-37.png)

**表 37.** CCWSC の例。
