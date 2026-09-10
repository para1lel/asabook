---
title: 'DeepSeek-V4.1-Flash: KV Cache Compression'
createTime: 2026/09/10 21:20:00
permalink: /ja/papers/deepseek-v4-1-flash/
---

> [DeepSeek-AI](https://www.deepseek.com/) が 2026 年 9 月 10 日に公開。本ウェブ閲覧版は公式技術報告 [DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf) に基づく。arXiv 登録および TeX ソースは公開されていないため、正確な文言、印刷レイアウト、完全な参考文献については <a href="/paper/deepseek-v4-1-flash.pdf" target="_blank" rel="noopener noreferrer">原 PDF</a> を正本とする。

## 概要

長期ホライズンのエージェントが広く利用されるようになり、モデルのワークロードは入力偏重になりつつある。従来研究によって長文脈計算のコストは大幅に削減されたものの、プリフィルは依然として計算量が大きく、大容量の KV キャッシュは HBM と SSD の容量、さらにデータ転送帯域を圧迫する。これらの計算・保存・帯域要求が、デプロイコストをさらに下げるうえで最大のボトルネックとなる。本稿では、552B のバックボーンパラメータを持ち、最大 100 万トークンの文脈を扱えるマルチモーダル Mixture-of-Experts（MoE）モデル DeepSeek-V4.1-Flash を提案する。Causal Encoder-Decoder（CED）により、デコード時にはトークン当たり 16B、プリフィル時には 8B のパラメータだけを活性化し、エージェント型ワークロードの費用効率を大幅に高める。KV キャッシュ圧縮の限界を押し広げるため、Compressed Sparse Attention 2（CSA2）でのレイヤー間 KV 再利用と FP4 KV キャッシュを組み合わせた。これにより、常時 HBM に置かれるグローバル KV キャッシュはトークン当たり 890 バイトとなり、DeepSeek-V4-Flash の約 4 分の 1 になる。さらに SWA Bounded Replay という専用のデプロイ最適化により、常時 SSD またはホストメモリに置かれる永続 KV キャッシュは約 8 分の 1 まで縮小する。KV キャッシュを大幅に減らしながら、モデル性能はベースラインを明確に上回る。また、DeepSeek-V4 の構成を簡素化し、複数の効率的なアーキテクチャ拡張も導入した。45T トークンからなるマルチモーダルコーパスで事前学習し、包括的な事後学習を行うことで、テキストおよびマルチモーダルの多様なエージェント場面で高い性能を得た。モデルのチェックポイントは [https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) で公開している。

<span id="figure-01"></span>

![図 1。 （a）エージェントベンチマークにおける DeepSeek-V4.1-Flash と比較対象の性能。（b）DeepSeek モデル各世代のトークン当たりグローバル KV キャッシュサイズ（バイト）。文脈メモリ要件を減らし続けてきた DeepSeek の取り組みを示す。DeepSeek-V4.1-Flash は、DeepSeek-V4-Flash および DeepSeek-V1 と比べ、トークン当たりのグローバル KV キャッシュをそれぞれ約 4 分の 1、437 分の 1 に削減した。](../../papers/deepseek-v4-1-flash/figure-01.png)

**図 1。** （a）エージェントベンチマークにおける DeepSeek-V4.1-Flash と比較対象の性能。（b）DeepSeek モデル各世代のトークン当たりグローバル KV キャッシュサイズ（バイト）。文脈メモリ要件を減らし続けてきた DeepSeek の取り組みを示す。DeepSeek-V4.1-Flash は、DeepSeek-V4-Flash および DeepSeek-V1 と比べ、トークン当たりのグローバル KV キャッシュをそれぞれ約 4 分の 1、437 分の 1 に削減した。

<span id="section-1"></span>

## 1 はじめに

近年、長期ホライズンのエージェント利用が急速に広がり、超長文脈処理はますます重要なモデルワークロードとなっている。このようなワークロードを支えるには、長い系列を効率よく処理するだけでなく、大規模な KV キャッシュを永続化し、再利用し、転送する必要もある。したがって KV キャッシュ管理はモデルデプロイの基盤機能となった一方、計算、保存、通信の各面に大きな課題をもたらした。スパースアテンションに関する従来の進展 [Dee25a, Dee26] によって長系列処理の計算コストは大幅に低下し、その分、永続化とデータ移動がより顕著なボトルネックとなっている。

具体的には、DeepSeek-V4 [Dee26] は、文脈全体を対象とするグローバルアテンション分岐とローカルな Sliding-Window Attention（SWA）を組み合わせる。グローバル分岐は main KV と indexer K からなる global KV を保持し、SWA はローカル KV 状態を保持する。ウィンドウサイズが固定なら、SWA KV の保存量は系列長に依存せず上限がある。したがって系列が十分長い場合、実行時 KV の大部分は global KV となり、その容量は HBM に制約される。また、接頭辞再利用のために一部の KV を永続 KV キャッシュとして保存するが、こちらは SSD とホストメモリ容量に制約される。I/O と相互接続帯域も、キャッシュの移動と読み込みを制限する。これらの制約は総じてサービングのスループットを抑え、デプロイコストを増やし、より長いタスクホライズンとより幅広い応用場面へのエージェント展開・普及を妨げる。

したがって、KV キャッシュの占有量をさらに減らすことは、保存・通信のボトルネックを緩和し、長文脈サービングのコストを下げるうえで不可欠である。そこで、より大胆な KV キャッシュ圧縮を目指したマルチモーダル Mixture-of-Experts（MoE）モデル DeepSeek-V4.1-Flash を開発した。DeepSeek-V4.1-Flash は 552B のバックボーンパラメータを持ち、マルチモーダル入力をネイティブに扱い、最大 100 万トークンの文脈に対応する。Causal Encoder-Decoder（CED）アーキテクチャを採用し、デコーダの global KV をエンコーダ最終隠れ状態から射影する。この設計により、トークン当たりの活性化パラメータはプリフィル時 8B、デコード時 16B となり、入力偏重のエージェント場面で特に費用効率が高い。DeepSeek-V4-Flash よりかなり大きいにもかかわらず、同じ系列長では実行時 KV キャッシュの保存量が約 4 分の 1、永続 KV キャッシュが約 8 分の 1 で済む。しかも総合性能は DeepSeek-V4-Flash を上回る。この圧縮率は、モデルアーキテクチャ、キャッシュ精度、デプロイ戦略を共同で最適化して達成した。概念的には、DeepSeek-V4 は SWA ベースのローカル処理バックボーンに、圧縮したグローバル文脈を付加したものと捉えられる。この見方から、ローカルアテンション設計をほぼ維持しつつ、グローバル分岐の簡素化に注力した。アーキテクチャ面では Compressed Sparse Attention 2（CSA2）を設計し、main KV と indexer K を含む global KV、および Top-K インデックスをレイヤー間で再利用して、KV キャッシュ保存量を大きく減らす。CSA2 には静的に割り当てる Full、Reindex、Reuse の 3 モードがある。Full Mode は global KV を生成してインデックス付けを行う。Reindex Mode は前段レイヤーの global KV を再利用し、自身の indexer Q で共有 indexer K を再スコアリングして新しい Top-K インデックスを選ぶ。Reuse Mode は前段レイヤーの global KV と Top-K インデックスの両方を再利用し、そのままスパースアテンションを実行する。どのモードでも、各レイヤーは固有の global Q と SWA KV を持つ。global KV と indexer K の共有により重複保存を減らせる。また、CSA と Heavily Compressed Attention（HCA）の混成アーキテクチャを使う DeepSeek-V4 と異なり、DeepSeek-V4.1-Flash は純粋な CSA2 を用いる。キャッシュ精度については、学習中に FP4 global KV キャッシュを使いながら、性能低下をわずかに抑えた。CSA2 と FP4 KV キャッシュを合わせることで、[図 1（b）](#figure-01)に示すように、global KV キャッシュ保存量は DeepSeek-V4-Flash の約 4 分の 1 になる。デプロイ面では DeepSeek-V4 と同じく、全レイヤーで Sliding-Window Attention（SWA）を使う。DeepSeek-V4 では、SWA KV キャッシュ永続化の保存コストと、厳密な再構築に要する計算との均衡を取る混成戦略を用いた。厳密な再構築には、直近の $L\times n_{\mathrm{win}}$ トークンを再生する必要がある。ここで $L$ はレイヤー数、$n_{\mathrm{win}}$ は SWA のウィンドウサイズである。DeepSeek-V4.1-Flash では SWA Bounded Replay を導入し、直近の $n_{\mathrm{win}}$ トークンだけを再生して必要な SWA KV 状態を近似的に再構築する。

実験では、これによる性能低下は無視できるほど小さかった。この結果から新たな保存量と計算量のトレードオフが得られ、少量のプリフィル再計算と引き換えに、SWA KV キャッシュを SSD に永続化せずに済む。SWA Bounded Replay によって、永続 KV キャッシュの占有量は DeepSeek-V4-Flash の約 8 分の 1 までさらに減る。これらの最適化により HBM と SSD の容量圧力を大幅に緩和し、デプロイコストを下げ、より大規模な展開への道を開く。

CED と CSA2 に加えて、元の DeepSeek-V4 アーキテクチャもさらに簡素化した。また、従来の mHC [Xie26] を Single-Pass mHC に更新し、付随するデプロイ用 Mega-mHC カーネルにより、従来の 4 カーネル実装と比べて活性化メモリトラフィックを半減させた。さらに、条件付きメモリモジュール Engram [Che26b] を組み込み、モデル能力を強化した。半自己回帰的なドラフト生成と、信頼度に応じた検証スケジュールを組み合わせる投機的デコードアーキテクチャ DSpark [Che26c] も導入した。これらすべての改善を組み合わせても、DeepSeek-V4.1-Flash の 1 トークン当たり Decode FLOPs は文脈長によらずほぼ一定である。[図 2](#figure-02)に示すように、文脈長を 4K から 1M へ 256 倍に伸ばしても Decode FLOPs の増加は 4 分の 1 にすぎず、DeepSeek-V4-Flash の増加幅を大きく下回る。

<span id="figure-02"></span>

![図 2。 DeepSeek モデル各世代における、文脈長と 1 トークンの Decode FLOPs の関係。計算精度を考慮し、BF16、FP8、FP4 の演算をそれぞれ 1、0.5、0.25 で重み付けした。DeepSeek-V4.1-Flash は文脈長が増えても Decode FLOPs がほぼ一定であり、長文脈場面の計算コストを大幅に減らす。](../../papers/deepseek-v4-1-flash/figure-02.png)

**図 2。** DeepSeek モデル各世代における、文脈長と 1 トークンの Decode FLOPs の関係。計算精度を考慮し、BF16、FP8、FP4 の演算をそれぞれ 1、0.5、0.25 で重み付けした。DeepSeek-V4.1-Flash は文脈長が増えても Decode FLOPs がほぼ一定であり、長文脈場面の計算コストを大幅に減らす。

これらのアーキテクチャがもたらす KV キャッシュ圧縮の利点を余すところなく引き出し、学習と推論の効率をさらに高めるため、DeepSeek-V4.1-Flash の学習基盤と推論システムを体系的に協調最適化し、効率的でスケーラブルな大規模マルチモーダル学習と長文脈デプロイを実現した。学習基盤は、ビジョンエンコーダの分離実行、長系列に対する画像の均衡シャーディング、アテンション再利用に向けたステージ間共有状態管理を支える。推論システムは Encoder と Decoder の SWA Bounded Replay 経路を実装する。ほかにも通信と計算のオーバーラップ、Engram 埋め込みテーブルのシャーディング、推論カーネル融合を行った。とりわけ CSA2 の Reuse Mode レイヤーは、プリフィル時 15 カーネル、デコード時 11 カーネルだけで実行できる。また、ホストメモリ上で長寿命の global KV と短寿命のエンコーダ SWA KV を分離し、Bounded Replay で欠落したエンコーダ SWA 状態を近似的に再構築する。

事前学習では、45T トークンからなる大規模マルチモーダルコーパスで DeepSeek-V4.1-Flash を学習した。密アテンションのウォームアップ段階を設けず、系列長 64K でスパースアテンションを最初から学習する。事前学習後のモデルはマルチモーダル能力をネイティブに持ち、最大 100 万トークンの文脈に対応する。評価では、DeepSeek-V4.1-Flash-Base は DeepSeek-V4-Pro-Base に匹敵する世界知識、推論、コーディング能力を示し、総パラメータは 3 分の 1、活性化パラメータは 4 分の 1 にとどまりながら、留保評価では 5%-10% 向上した。これらの結果は高いパラメータ効率を示すとともに、実運用に向けた学習データ品質の改善を反映している。

このベースモデルを基に、推論能力とエージェント能力を引き出す事後学習を行った。前述のアーキテクチャ上の革新とは対照的に、事後学習にはアルゴリズム上の新規性はない。レシピは教師ありファインチューニング（SFT）の後に強化学習（RL）とオンポリシー蒸留（OPD）を行う標準形に従い、DeepSeek-V4 の開発で使った確立済みの手法 [Dee26] 以外に変更を加えていない。本質的な変更はすべてデータパイプラインにある。データ合成と環境構築の大規模な自動化パイプラインを開発し、RL に用いるデータ、タスク、ロールアウトを段階的に拡大することで、テキスト、マルチモーダル、エージェントの各領域へモデル能力を広げた。[図 1（a）](#figure-01)は、主要なエージェントベンチマークにおける DeepSeek-V4.1-Flash の性能をまとめたものである。評価から、コンパクトな規模でありながら、DeepSeek-V4.1-Flash は次のような特徴的な能力構成を示した。

- **推論。** 数学や競技プログラミングなど推論負荷の高いベンチマークで高精度を維持し、Kimi-K3 [Kim26c] や DeepSeek-V4-Pro といった最上位のオープンソースモデルに匹敵する性能を示す。

- **エージェント。** Terminal-Bench 2.1 [Mer26]、DeepSWE v1.1 [Dee26c]、AutomationBench [She26] など標準的なエージェントベンチマークで、クローズドソースの最先端モデルと同等の性能を達成した。日常のコーディングタスクとホワイトカラー業務フローを十分にこなせる。ただし、専門家水準の領域知識を要する Terminal-Bench 4.0 [Mar26] のような科学系エージェントタスクでは、巨大モデルとの差が残る。

- **マルチモーダル。** マルチモーダル領域では、視覚推論と専門的な図表の解釈を測るベンチマークにおいて、Kimi-K3 など最上位のオープンソース競合を上回る。正式な指標に加え、フロントエンド開発やオフィス自動化など現実の視覚エージェント業務でも、レンダリングした画面キャプチャを利用して目視検査と自己修正を行う実用性を示した。それでも、巨大なクローズドソースシステムとの比較では総合性能に明確な差が残ることを認める。

これらの結果から、DeepSeek-V4.1-Flash は大多数のベンチマークですでにクローズドソースの最先端モデルに匹敵し、現実のタスクの 95% 超を完了できることが分かる。同時に、活性化フットプリントが小さいため、推論レイテンシとサービングコストも低い。したがって DeepSeek-V4.1-Flash は、能力と効率のバランスに優れ、幅広い利用者の日常業務を支える高速で手頃なアシスタントになり得る。要するに、DeepSeek-V4.1-Flash はデプロイコストを下げながら、モデルの知能と推論効率を同時に高める。長期ホライズンのエージェントを大規模展開する際のコスト障壁を大きく下げ、より広い場面への導入機会を生み出す。また、今後のスケーリングに向けた新たな出発点でもある。この基盤の上で、モデルアーキテクチャ、事前学習、事後学習を共同でスケールし、モデル知能の最前線をさらに探究していく。

<span id="section-2"></span>

## 2 アーキテクチャ

<span id="section-2-1"></span>

### 2.1 全体像

DeepSeek-V4.1-Flash は、画像とテキストを入力としてテキストを自己回帰的に生成するマルチモーダル Mixture-of-Experts（MoE）Transformer である。言語バックボーンは 40 層の causal Transformer からなり、20 層の causal encoder と、それに続く 20 層の decoder で構成される。最初の 2 層だけは SWA のみを使い、それ以外の各層は global attention と sliding window attention（SWA）の両方を備える。vision encoder と MLP projector が画像を visual embedding に変換し、text embedding と共同で処理する。マルチモーダルデータは言語モデル事前学習の最初から取り込む。全体として DeepSeek-V4.1-Flash は 552B のバックボーンパラメータと 196B の Engram パラメータを持ち、トークン当たりの活性化パラメータはプリフィル時 8B、デコード時 16B である。[図 3](#figure-03)に全体構成を示す。

<span id="figure-03"></span>

![図 3。 DeepSeek-V4.1-Flash の全体アーキテクチャ。40 層ネットワークを 20 層ずつの causal encoder と decoder に分ける。すべてのフィードフォワード層には標準の DeepSeekMoE を用いる。encoder の最初の 2 層は sliding window attention（SWA）、残りは Compressed Sparse Attention 2（CSA2）を用い、CSA2（ratio, mode）は圧縮率とモードを表す。Single-Pass mHC、Engram、DSpark、Hierarchical Sparse Indexer も使用する。](../../papers/deepseek-v4-1-flash/figure-03.png)

**図 3。** DeepSeek-V4.1-Flash の全体アーキテクチャ。40 層ネットワークを 20 層ずつの causal encoder と decoder に分ける。すべてのフィードフォワード層には標準の DeepSeekMoE を用いる。encoder の最初の 2 層は sliding window attention（SWA）、残りは Compressed Sparse Attention 2（CSA2）を用い、CSA2（ratio, mode）は圧縮率とモードを表す。Single-Pass mHC、Engram、DSpark、Hierarchical Sparse Indexer も使用する。

Causal Encoder-Decoder（CED）と Compressed Sparse Attention 2（CSA2）は、長文脈推論における相補的なコストに対処する。CED は encoder 出力から decoder の global key-value（KV）キャッシュを構築し、各層固有の sliding-window attention を保ちながら、プロンプトの大半のトークンが decoder の全計算を通らずに済むようにする。これによりプリフィル計算はほぼ半減し、文脈が伸びるエージェントワークロードで新規または未キャッシュ入力を処理するコストが下がる。CSA2 は global KV を層間共有してキャッシュ保存量を減らし、スパース選択を再利用してインデックス処理を減らす。decoder では Hierarchical Sparse Indexer が、後段の indexer の探索対象を前段 indexer が選んだ候補プールに限定し、クエリ当たりのスコアリング項目数をさらに減らす。

DeepSeekMoE [Dai24] の共有 expert と細粒度 routed expert を維持し、画像トークンとテキストトークンにモダリティ別 load balancing [Wan24d] を導入する。Single-Pass mHC [Xie26] は residual stream の混合を見直して効率的なカーネル融合を可能にし、Engram [Che26b] はスパースアクセスされる条件付きメモリを加える。バックボーンの事前学習では MTP モジュールを省き、投機的デコードには DSpark [Che26c] を使う。DSpark はバックボーン事前学習の後に別途学習する。さらに main KV キャッシュを FP4 に圧縮し、保存オーバーヘッドをいっそう減らす。以下では、これらの構成要素と対応する最適化変更を説明する。

<span id="section-2-1-1"></span>

#### 2.1.1 マルチモーダルアーキテクチャ

マルチモーダル入力経路は vision encoder と MLP projector からなる。各入力画像について、vision encoder は空間的な visual feature グリッドを生成する。続いて 3 × 3 の pixel-unshuffle が各局所近傍をチャネル次元に沿って並べ替え、空間解像度を下げてから、MLP projector が feature を言語バックボーンの hidden dimension に写像する。最後に、得られた visual embedding を入力 embedding 系列内の対応する image-token 位置へ挿入し、言語バックボーンで text embedding と共同処理する。

**DeepSeek-ViT。** 可変解像度の画像をネイティブに処理する vision encoder DeepSeek-ViT をゼロから学習する。DeepSeek-ViT は Vision Transformer [Dos20] を基に、いくつかの変更を加えて構築した。任意解像度の入力に対応するため、標準の absolute positional embedding を 2D-RoPE に置き換える。ViT を LLM の設計原則へ近づけるため、patch embedding 層の畳み込みを線形射影に置き換え、Muon optimizer との互換性を確保する。正規化には RMSNorm [Zha19]、活性化関数には SwiGLU [Sha20] を採用する。visual feature を LLM に渡す前に、3 × 3 ダウンサンプリングの pixel-unshuffle を適用して visual token 数を 9 分の 1 に減らし、最大約 1344 × 1344 ピクセルの入力解像度を実質的に扱えるようにする。

**MoE のマルチモーダル補助損失なし負荷分散。** 画像トークンとテキストトークンでは表現分布が異なり、MoE の expert routing に対する選好も異なる可能性がある。両者をまとめて負荷分散すると、モダリティ固有の不均衡が見えにくくなり得る。そこで、補助損失なし負荷分散 [Wan24d] を拡張し、テキストトークンと画像トークンで expert ごとの補正バイアスを別々に保持する。routing 時には各トークンが自身のモダリティに対応する補正バイアスで expert を選択し、選ばれた expert 出力の重み付けには元の routing score を使う。各学習ステップ後、2 組のバイアスをそれぞれの expert load に従って独立に更新する。この設計はモダリティごとの expert 利用を均衡させ、安定かつ効率的なマルチモーダル学習に寄与する。

<span id="section-2-2"></span>

### 2.2 Causal Encoder-Decoder（CED）

エージェント型ワークフローでは頻繁なツール呼び出しによって大量のプリフィル要求が生じ、KV キャッシュミス時の計算負荷が深刻になる。このボトルネックを緩和するため、YoCo [Sun24b] に着想を得た Causal Encoder-Decoder（CED）を提案する。YoCo は上半分の層が下半分の層で生成した KV キャッシュを直接共有することで、プリフィル計算を減らす。この考えを発展させ、CED は KV キャッシュ全体の容量と KV 生成の計算深度をともに高める一連の構造改善を導入する。その結果、ベースラインと同等の性能を保ちながら、プリフィル計算をほぼ半減できる。

global attention では、CED は Transformer の下側 $L/2$ 層を causal encoder とみなす。上半分の層、すなわち decoder（$l>L/2$）では、KV entry を各層の hidden state $H_l$ から導出しない。代わりに、層ごとの射影重み（$W_l^{\mathrm{KV}}$ と $W_l^Z$）を使い、$L/2$ 番目の層の hidden state $H_{L/2}$ から直接射影する。

<span id="equation-01"></span>

$$
C_l=H_{L/2}W_l^{\mathrm{KV}},\qquad Z_l=H_{L/2}W_l^Z,\qquad l>\frac{L}{2},
$$


ここで $C$ と $Z$ は、それぞれ KV entry と対応する圧縮重みを表す。この設計により、CED はプリフィル段階で前半の層だけを計算し、最小限の計算コストで上層の global KV キャッシュを取得できる。

sliding window attention（SWA）については、CED は全層で従来どおりの層ごとの計算を維持する。具体的に、任意の層 $l$ の local key と value は、その層の hidden state $H_l$ から直接導出する。この設計により local KV 生成の計算深度は実質的に増える。ただし、層ごとの計算を維持するには SWA replay が必要になる。プリフィル段階で decoder の SWA KV キャッシュを計算するには、追加で $n_{\mathrm{win}}\times L/2$ トークンを処理する必要がある。ここで $n_{\mathrm{win}}$ はウィンドウサイズを表す。1 ターン当たりのプロンプトが短いマルチターン対話では、decoder のこの計算負荷は無視できない。幸い、従来研究 [Che25ad] から、SWA の実効 receptive field は理論上の $n_{\mathrm{win}}\times L/2$ よりはるかに小さいことが分かっている。この観察に基づき、SWA 計算ではプロンプト末尾の $n_{\mathrm{win}}$ トークンだけをプリフィルする Decoder SWA Bounded Replay を導入し、計算コストを大きく減らす。詳細は[第 3.2.2 節](#section-3-2-2)で述べる。

全体として、系列長 $N\gg n_{\mathrm{win}}$ に対し、CED はプリフィル計算量を $O(N\,L)$ から $O(N\,L/2+n_{\mathrm{win}}\times L/2)\approx O(N\,L/2)$へ減らし、総計算量を実質的に半減させる。

<span id="section-2-3"></span>

### 2.3 Compressed Sparse Attention 2（CSA2）

長文脈をサービングするには、KV キャッシュ保存量とアテンション計算量の両方を抑える必要がある。これらのコストは、乗算的に効く 3 つの次元で削減できる。entry size の次元では、GQA [Ain23] が KV head 数を減らし、MLA [Dee24] が head 間で小さな latent を共有する。sequence の次元では、DeepSeek-V4 [Dee26] の CSA や HCA のように、$m$ トークンごとに 1 entry へ圧縮する。layer の次元では、一部の層が自身のキャッシュを保持せず他層のキャッシュ [Bra24] や選択結果を再利用するか、層自体をより効率的なものへ置き換える。従来研究は layer 次元での圧縮が有効だと示している。IndexCache [Bai26] は Top-K インデックスを層間で再利用して indexer 計算を減らし、YOIO [Sun26b] はスパース routing を一度だけ計算して全層で共有し、HySparse [Gao26] は sparse layer が dense layer の KV キャッシュを再利用する。しかし、インデックス再利用だけでは main KV の保存量は減らず、ネットワーク全体での routing 共有は性能を制限し、混成設計にも full attention layer が残る。さらに重要なのは、いずれも 3 つの乗算的次元すべてを対象としていないことである。

CSA2 は 3 次元を同時に活用する。main KV と indexer K を層間共有し、Top-K インデックスを層間再利用できるようにする一方、キャッシュ共有とインデックス再利用は分離する。これらの再利用戦略を、単純化した compressor、および decoder の後段 indexing layer の探索範囲を狭める Hierarchical Sparse Indexer と組み合わせる。CSA と同様、CSA2 には indexer Q と indexer K で main KV entry をスコアリングし、各 query に対する Top-K entry を選ぶ軽量 indexer がある。各 Q は、選択した entry と層固有の sliding-window KV（SWA KV）をまとめて attend する。CSA2 は圧縮率 1 の非圧縮 main KV も特殊ケースとして含む。同時に、compressor と indexer の両方を簡素化する。CSA では、圧縮率 $m$ のとき $2m$ 個の元 KV cache entry から各 main KV entry を生成し、隣接する圧縮 entry どうしで元 entry が重複する。また、圧縮時に $2m$ entry の位置を符号化する absolute positional embedding を含む。CSA2 はこの重複と absolute positional embedding を除く。さらに CSA2 は main KV entry を射影して indexer K を得るため、hidden state からの独立した圧縮経路を持つ CSA を置き換える。どちらの設計変更も実装を単純にし、学習効率を高める。[第 2.3.1 節](#section-2-3-1)と[第 2.3.2 節](#section-2-3-2)で、層間再利用戦略と Hierarchical Sparse Indexer をそれぞれ説明する。

<span id="section-2-3-1"></span>

#### 2.3.1 層間 KV・インデックス再利用

各 CSA2 層には Full、Reindex、Reuse のいずれか 1 モードを静的に割り当てる。3 モードすべてで、層は自身の query と SWA KV を計算し、選択した main KV entry と合わせて新しい attention output を生成する。モード間で異なるのは、main KV、indexer K、Top-K インデックスの取得方法である。[図 4](#figure-04)に 3 モードを示す。

<span id="figure-04"></span>

![図 4。 CSA2 の 3 つの動作モード。main KV、indexer K、Top-K インデックスの取得方法が異なる。緑のブロックは現在の層で計算する量、黄のブロックは直近の Full Mode 層から再利用する main KV と indexer K、赤のブロックは直近のインデックス生成層（Full または Reindex Mode）から再利用する Top-K インデックスを示す。3 モードすべてで main Q と SWA KV は現在の層で計算する。](../../papers/deepseek-v4-1-flash/figure-04.png)

**図 4。** CSA2 の 3 つの動作モード。main KV、indexer K、Top-K インデックスの取得方法が異なる。緑のブロックは現在の層で計算する量、黄のブロックは直近の Full Mode 層から再利用する main KV と indexer K、赤のブロックは直近のインデックス生成層（Full または Reindex Mode）から再利用する Top-K インデックスを示す。3 モードすべてで main Q と SWA KV は現在の層で計算する。

**Full Mode。** 層は自身の main KV と indexer Q を計算し、その main KV から indexer K を射影して indexer を実行し、新しい Top-K インデックスを生成する。したがって CSA2 の完全な計算経路を実行し、DeepSeek-V4 の完全な CSA 層と同じ構成要素を担う。

**Reindex Mode。** 層は、前段層から得られる直近の main KV と、それに対応する indexer K を再利用する。indexer は自身の query を計算し、再利用した key を再スコアリングして、新しい Top-K インデックスを生成する。これにより main KV と indexer K を共有したまま、スパース選択を層ごとに変えられる。

**Reuse Mode。** 層は、直近の main KV と、その main KV に対して前段の Full または Reindex Mode 層が計算した最新の Top-K インデックスを再利用する。indexer Q の計算も index score の評価も行わず、この選択を用いてアテンションを実行する。

main KV と indexer K の共有はキャッシュ保存量を減らし、Top-K インデックスの再利用は追加の indexer 計算をなくす。Reindex Mode ではキャッシュ共有を保ちながら、選択 entry を層ごとに変えられる。CSA2 と CED を組み合わせる場合、Full Mode を割り当てた decoder 層は、causal encoder の最終層に当たる$L/2$ 番目の層の hidden state から自身の global KV を計算する。Reindex と Reuse Mode は変わらない。

<span id="section-2-3-2"></span>

#### 2.3.2 Hierarchical Sparse Indexer

層間のインデックス再利用は indexer の評価回数を減らすが、残る indexer はなお因果的に可視な文脈全体をスコアリングする。文脈が極端に長い場合、このコストは依然として大きな計算ボトルネックとなる。従来研究では、token-level indexing の前に pooled block representation をスコアリングして枝刈りすることで indexer sparsity を導入した [Xu26]。decoder では、浅い indexer の情報を使って、追加状態なしに深い indexer が検討する候補を自然に制限できることが分かった。そこで CED の decoder のみで使う Hierarchical Sparse Indexer を導入し、decode 時の反復スコアリングを減らす。各 query に対して、Full Mode を割り当てた最初の層が候補プールを構築し、後続の re-indexing layer はそれを探索範囲として使う。候補プールサイズを固定すると、深い indexer の query 当たりコストは文脈長に対する線形から定数へ変わる。この仕組みは学習を考慮して事後学習時に導入する。候補制限を学習と推論で同じように適用するため、深い indexer は推論時と同じ探索範囲のもとで最適化される。[図 5](#figure-05)にこの処理を示す。最初の Full Mode 層は、因果的に可視な全 main KV position をスコアリングし、自身のアテンション用 Top-K インデックスを生成する。同時に blockwise candidate selection も行う。各 block には内部 position の最大 index score を割り当て、score が最も高い block を選ぶ。次に選択 block が覆う position を、最終 Top-K より大きな候補プールへ集める。たとえば 8 position の block を 2,048 個選ぶと、16,384 個の候補 position が得られる。このプールが後続 indexer の探索範囲を定め、最終 Top-K 選択が各層の読み込む main KV entry を決める。

<span id="figure-05"></span>

![図 5。 Hierarchical Sparse Indexer。各正方形が 1 position を表し、緑の正方形は選択したインデックス、青い長方形は最大 indexer score に基づいて選んだ block を示す。decoder の最初の Full mode CSA2 層が自身の Top-512 インデックスを選び、選択 block から後続層向けの共有候補プールを構築する。その後 Reindex mode の CSA2 層は、このプールから Top-512 インデックスを選ぶ。](../../papers/deepseek-v4-1-flash/figure-05.png)

**図 5。** Hierarchical Sparse Indexer。各正方形が 1 position を表し、緑の正方形は選択したインデックス、青い長方形は最大 indexer score に基づいて選んだ block を示す。decoder の最初の Full mode CSA2 層が自身の Top-512 インデックスを選び、選択 block から後続層向けの共有候補プールを構築する。その後 Reindex mode の CSA2 層は、このプールから Top-512 インデックスを選ぶ。

後続の Reindex Mode 層は、対応する query の候補 position だけをスコアリングし、そのプール内で自身の Top-K entry を選ぶ。Reuse Mode 層は新たな indexing を行わず、再利用する main KV に対して計算済みの最新 Top-K インデックスを使う。このように候補プールは indexing layer 間で共有するが、最終選択は異なってよい。

候補プールサイズが固定なら、後続の各 indexer が query 当たりにスコアリングする position 数は文脈長に依存せず上限がある。最初の Full Mode 層だけは因果的に可視な範囲全体を走査する。したがって hierarchical indexing は、初回の全範囲走査を保ちながら、後続 indexer 評価のコストを減らす。

<span id="section-2-4"></span>

### 2.4 効率的なアーキテクチャ拡張

<span id="section-2-4-1"></span>

#### 2.4.1 Single-Pass mHC

DeepSeek-V4 では、隣接する Transformer block 間に $n$ 本の residual stream を保持する mHC [Xie26] を導入した。各 token についてこれらを $X_l\in\mathbb{R}^{n\times d}$ と表す。$l$ は block index、$d$ は hidden dimension である。stream は次のように更新する。

<span id="equation-02"></span>

$$
X_{l+1}=B_lX_l+C_l\mathcal{F}_l(A_lX_l),\qquad (A_l,B_l,C_l)=\mathcal{H}(X_l),
$$

ここで $A_l\in\mathbb{R}^{1\times n}$、$C_l\in\mathbb{R}^{n\times1}$、$B_l\in\mathbb{R}^{n\times n}$ は、$X_l$ から予測する token-wise coefficient である。coefficient predictor $\mathcal{H}$ は正規化と射影を含む。

理想的には、2 つの block 間の residual transformation は$(X_{l-1},Y_{l-1})$ から $(X_l,\hat{X}_l)$ への単一 map となる。ここで $\hat{X}_l=A_lX_l$ は現在 block の入力、$Y_{l-1}=\mathcal{F}_{l-1}(\hat{X}_{l-1})$は前 block の出力である。この map には$(n+1)d$ 回の read と$(n+1)d$ 回の write が必要で、activation memory traffic の下限は$(2n+2)d$ となる。実際の DeepSeek-V4 は、データ依存により順次実行される 3 kernel を使い、式（2）を multi-pass で実装する。

<span id="equation-03"></span>

$$
X_l=B_{l-1}X_{l-1}+C_{l-1}Y_{l-1}\qquad\text{Residual update, contraction over }n
$$

<span id="equation-04"></span>

$$
(A_l,B_l,C_l)=\mathcal{H}(X_l)\qquad\text{Coefficients, contraction over }nd
$$

<span id="equation-05"></span>

$$
\hat{X}_l=A_lX_l\qquad\text{Input mixing, contraction over }n
$$


3 kernel はそれぞれ$(n+1)d$、$nd$、$nd$ 個の値を read し、合計$(n+1)d$ 個を書き込む。$\mathcal{F}_l$ の pre-norm を含めると、activation memory traffic は合計$(4n+4)d$ となり、下限の 2 倍である。

$\mathcal{H}$ では、正規化重みを事前に射影重みへ畳み込み、射影後に RMS 除算を適用する。このため 3 段階のうち 2 段階は residual の 1 回の走査を共有できる。residual update は hidden dimension にわたる reduction を要しないので、$X_l$ の各 tile を計算後すぐに使い、射影出力と RMS 計算に必要な平方和を累積できる。$A_l$ は全 hidden tile の reduction が完了するまで得られないため、input mixing をこの pass に融合することはできない。そのため $X_l$ をもう一度 read する必要がある。この 2 回目の pass には input pre-norm も組み込める。この two-pass 実装の activation read/write は合計$(3n+2)d$ となり、下限より $X_l$ の read が 1 回多い。

そこで Single-Pass mHC を導入し、input-mixing coefficient を 1 block ずらす。つまり各 block が前 block の生成した mixing coefficient を使うことで、上記の依存をなくす。

<span id="equation-06"></span>

$$
X_{l+1}=B_lX_l+C_l\mathcal{F}_l(A_{l-1}X_l),\qquad (A_l,B_l,C_l)=\mathcal{H}(X_l).
$$

input mixing は $A_l$ ではなく $A_{l-1}$ を使うため、$X_l$ から計算する coefficient に依存しなくなる。したがって $X_l$ の各 tile は、全 reduction の完了を待たず、input mixing と coefficient prediction の両方にすぐ使える。実験上、このずらしによる性能低下は無視できるほど小さい。

事前学習では、ずらしによって各 block が使う mixing coefficient だけが変わるため、従来の multi-kernel 実装を維持する。デプロイでは residual update、input mixing、coefficient prediction を単一 kernel Mega-mHC に融合する。この kernel は、mHC を$(3n+2)d$ 回、Single-Pass mHC を$(2n+2)d$ 回の activation read/write で実装する。Mega-mHC は hidden dimension 方向に $X_l$ を tile 分割して処理する。各 tile を使って mixed input を計算し、次 block の（$A_l$, $B_l$, $C_l$）を予測するための量を累積する。input pre-norm と FP8 変換も kernel に組み込む。これにより residual を 1 回 read、1 回 write するだけで、理想 map の$(n+1)d$ read と$(n+1)d$ write を達成し、従来実装の activation memory traffic を半減させる。

<span id="section-2-4-2"></span>

#### 2.4.2 Engram

記憶と計算を分離するため以前の研究で導入した条件付きメモリモジュール Engram [Che26b] を DeepSeek-V4.1-Flash に追加する。tokenizer compression、multi-head hashing、context-aware gating、multi-branch integration という元の Engram 設計に従いつつ、2 点を変更する。第 1 に、性能向上が推論スタックの複雑化に見合わないため、短い causal convolution を省く。第 2 に、[第 2.5 節](#section-2-5)で述べるように、momentum-based update の後に Sinkhorn balancing を行い、Engram embedding を最適化する。

196B の Engram パラメータを 2 モジュールへ均等に割り当てる。各モジュールは $N$-gram order {2, 3, 4} を用い、hash head は 8 個、order 当たりの総 embedding dimension は 2048 である。各 head は約 16M entry のテーブルを索引し、テーブルサイズには互いに異なる素数を選ぶ。embedding table と key/value projection はともに FP8 精度を使う。学習 pipeline stage 間のメモリ使用量を均衡させるため、モジュールは layer 1 と 14（0 始まり）に置く。推論時には deterministic addressing により、バックグラウンド RDMA 転送でホストメモリから embedding を先読みできる。第 1 モジュールの先読みは、最初の Transformer block の計算と重ねる。Engram の学習・推論実装の詳細は[第 3.1.3 節](#section-3-1-3)で述べる。

<span id="section-2-4-3"></span>

#### 2.4.3 DSpark

DeepSeek-V4.1-Flash に、semi-autoregressive drafting と confidence-scheduled verification を組み合わせた投機的デコードモジュール DSpark [Che26c] を搭載する。

drafter は sliding attention window 128 token の Transformer block 3 層からなる。これらを 1 回 forward すると、5 個の draft position の base logit を並列計算し、軽量 Markov head が draft token 間の依存をモデル化する。confidence head は position ごとの条件付き accept probability を予測し、それを使って prefix survival probability を推定する。scheduler はこの推定値と事前計測した engine throughput curve を組み合わせ、現在の system load のもとで期待 system-wide token throughput を最大化するよう、request ごとの verification length を動的に選ぶ。

事前学習を通じてバックボーンと共同学習する DeepSeek-V3 [Dee24a] の MTP モジュールとは異なり、DSpark は事前学習後の専用段階で導入する。この段階ではバックボーンを凍結し、DSpark だけを学習する。事後学習中は DSpark 目的関数の勾配をバックボーンへ伝播させず、バックボーンと並行して DSpark の学習を続ける。これにより DSpark を変化する policy と揃え、online serving と RL・OPD の rollout generation の両方を高速化できる。

<span id="section-2-4-4"></span>

#### 2.4.4 FP4 Main KV キャッシュ

長文脈のエージェントワークロードでは request ごとに大規模な KV キャッシュが必要となり、サービングコストが増える。

DeepSeek-V4 はすでに FP4 の indexer query と key に量子化認識学習（QAT） [Jac18] を用い、index 計算を高速化して indexer cache size を減らしている。実験では別形式の方が高精度だったが、可能な限り多くの hardware platform に対応するため、OCP 標準の MXFP4 形式 [Dar23] を採用した。今回 QAT を main KV キャッシュへ拡張する。ここでは FP4 は行列積の高速化ではなく保存量削減に使う。attention 前に cached value を dequantize すれば、その形式に対する native matrix-multiplication support を要求せずに、より高精度な形式を使えるため、hardware platform 間の互換性を維持できる。

評価した約 4-bit の形式から、NVFP4 [Alv25] に従い、16 channel ごとに 1 個の E4M3 scale を持つ E2M1 を選ぶ。ただし精度と単純さの均衡を取るため、第 2 段の global scale は省く。これを省いても main KV キャッシュには十分な dynamic range が残る。この形式は最大 $448\times6=2688$ の大きさを表せ、キャッシュの magnitude bound をはるかに上回る。DeepSeek-V4.1-Flash で学習済み RMSNorm weight の最大 magnitude は約 1 である。RMS normalization 後、512-channel KV latent の L2 norm は最大でも約 $\sqrt{512}$ である。RoPE はこの norm を保つため、回転後の channel 間最大 absolute value も約 $\sqrt{512}\approx22.6$ 以下となる。また、学習中に観測した最大 magnitude は約 10 だった。したがって global scale を省いても測定可能な精度低下はなく、cache layout を単純化できる。

DeepSeek-V4.1-Flash で FP4 main KV cache storage を使えるよう、事後学習に QAT を導入する。non-RoPE component と RoPE component は同じ量子化形式を使う。cache は RoPE 後に量子化する。RoPE 前の量子化は実験でごくわずかな精度向上しか得られず、decode 時に追加 overhead が生じるためである。SWA KV キャッシュは量子化に敏感なので FP8 を維持する。DeepSeek-V4 の FP8 main KV キャッシュと比べ、この形式は HBM 上でも SSD へ offload した場合でも保存量をほぼ半減させる。

<span id="section-2-5"></span>

### 2.5 最適化

DeepSeek-V4 で用いた最適化構成を基に、アーキテクチャ設計へより適合させるため、新たな変更をいくつか加えた。

第 1 に head-wise Muon を用い、Muon update の前に Query weight を head ごとに分割する。この設計の動機を簡単に説明する。Muon を preconditioned gradient descent とみなすと、vanilla Muon は全 head に 1 つの preconditioner を使う一方、head-wise Muon は head ごとに異なる preconditioner を与える。この設計は attention head 間の異質性をより適切に扱える [Zha24ad, Zha26b] [+zha26b-section-3]。その結果、head-wise Muon が vanilla Muon を上回ることを確認した。この実証的な優位性は GLM 5 [Zen26] と Kimi-K3 [Kim26c] でも検証されている。

[+zha26b-section-3]: [Zha26b] の三つ目の節。

第 2 に、新たに導入した Engram parameter へ Adam を適用すると optimizer-state のメモリ占有量が大幅に増える。学習時のメモリ使用量を抑えるため、代わりに momentum-based update の後に Sinkhorn balancing を行い、Engram embedding table、token embedding、prediction head を最適化する。Sinkhorn balancing は以前、SinkGD [Sce25] で linear-layer weight matrix に適用された。ここではこれを大規模 parameter matrix へ拡張する。Muon と同様に momentum buffer だけを必要とし、実験では Adam を上回った。

**基本構成。** normalization-layer weight と、bias や scaling factor を含むその他の non-matrix parameter には AdamW [Los17] を維持する。language-model backbone の linear transformation、Engram projection layer、vision-language projector の weight matrix には Muon [Kel24] を使う。Query と Key の weight には head-wise Muon を使う。Muon には decoupled weight decay と Nesterov momentum [Nes83, Liu25] を適用する。normalization-layer weight にも weight decay を適用するが、bias と scaling factor には適用しない。Sinkhorn-balanced update も Nesterov momentum を使うが weight decay は適用しない。事前学習では learning-rate decay 段階まで vision encoder を凍結する一方、最終 normalization layer と vision-language projector は学習可能に保つ。learning-rate decay の開始時に vision encoder の凍結を解除し、より小さい learning rate で LLM と共同最適化する。

**Engram／Embedding／Prediction Head の Sinkhorn-balanced update。** 手順全体を Algorithm 1 に示す。概略は Muon と同じ流れで、Newton-Schulz orthogonalization を Sinkhorn balancing に置き換える。大きい方の matrix dimension を $m$ とし、embedding table と prediction head では vocabulary size に対応する。hidden dimension を $n$ とする。

**アルゴリズム 1。Sinkhorn balancing を伴う momentum update**

- **必要条件：** 重み $W_t\in\mathbb{R}^{m\times n}$、勾配 $G_t$、momentum $M_{t-1}$、momentum coefficient $\beta$、base learning rate $\eta_t$、learning-rate correction $\gamma$、numerical constant $\varepsilon$ と $\tau$、奇数回の normalization step $K$。
- $M_t\leftarrow\beta M_{t-1}+(1-\beta)G_t$.
- $\hat{G}_t\leftarrow\beta M_t+(1-\beta)G_t$（Nesterov momentum）。
- $\rho_i\leftarrow\|\hat{G}_{t,i,:}\|_2$、$\bar{\rho}\leftarrow\frac{1}{m}\sum_{i=1}^{m}\rho_i$。
- $U^{(0)}\leftarrow\hat{G}_t$.
- $\rho_i\leq\tau\bar{\rho}$ の場合は $U^{(0)}_{i,:}\leftarrow0$ とする（near-zero row を mask）。
- $k=1,\ldots,K$ について**反復：**
  - $k$ が奇数なら、すべての $i=1,\ldots,m$ について：
    - $U^{(k)}_{i,:}\leftarrow U^{(k-1)}_{i,:}/(\|U^{(k-1)}_{i,:}\|_2+\varepsilon)$.
  - **それ以外は、**すべての $j=1,\ldots,n$ について：
    - $U^{(k)}_{:,j}\leftarrow U^{(k-1)}_{:,j}/(\|U^{(k-1)}_{:,j}\|_2+\varepsilon)$.
- $\Delta_t\leftarrow\sqrt{n}U^{(K)}$（unit row $\ell_2$ norm を unit row RMS に変換）。
- $\tilde{\eta}_t\leftarrow\gamma\eta_t$（update magnitude を Adam に合わせる）。
- $W_{t+1}\leftarrow W_t-\tilde{\eta}_t\Delta_t$.


Nesterov momentum update $\hat{G}_t$ が与えられたとき、Sinkhorn balancing は次を満たす対角 scaling matrix $D_r$ と $D_c$ を求める。

<span id="equation-07"></span>

$$
\begin{alignedat}{2}
\Delta_t &= \sqrt{n}U^{(K)} &&= \sqrt{n}D_r\hat{G}_tD_c,\\
\frac{1}{n}\sum_{j=1}^{n}(\Delta_t)_{ij}^{2} &&&\approx 1,\\
\frac{1}{m}\sum_{i=1}^{m}(\Delta_t)_{ij}^{2} &&&\approx 1,
\end{alignedat}
$$

したがって、この手順は update matrix の row-wise RMS と column-wise RMS を近似的に等しくする。ここで 1 row は 1 token index または n-gram identity に対応し、1 column は 1 hidden feature を符号化する。Sinkhorn balancing は row と column の両方で正規化し、この token-feature 構造を活用する。数値安定性のため、$\rho_i\leq\tau\bar{\rho}$ を満たす row は mask する。係数 $\sqrt{n}$ は unit row $\ell_2$ norm を unit row-wise RMS へ変換する。別途、Adam の update magnitude に合わせるため effective learning rate を $\tilde{\eta}_t=\gamma\eta_t$ と調整する。$\gamma=0.18$ とし、Moonlight [Liu25] で用いた係数 0.2 に近い。

より広く見ると、Sinkhorn balancing は matrix または tensor の axis structure を利用する optimizer [Sha18, Zha25ay, Wen25b, Gle25, Den26, Yua26a, Xu26a] と密接に関係する。たとえば Adafactor [Sha18] は別の方法で row-wise・column-wise normalization を行い、Adam-mini [Zha25ay] は embedding table と prediction head に別方式の row-wise normalization を使う。これらの normalization strategy は optimization performance と communication overhead が異なり得る。詳細な検討は今後の課題とする。

<span id="section-3"></span>

## 3 基盤システム

<span id="section-3-1"></span>

### 3.1 学習基盤

<span id="section-3-1-1"></span>

#### 3.1.1 マルチモーダル学習基盤

**対照学習における通信と計算のオーバーラップ。** vision encoder は、生成的な next-token prediction loss で fine-tune する前に、まず contrastive objective で最適化する。contrastive phase では text-vision pair の full batch に対して loss を計算するため、両 modality の feature を data-parallel rank 全体で all-gather する必要があり、通信量が大きい。text feature の gradient は集約済み visual feature のみに依存し、対称的に visual feature の gradient は集約済み text feature のみに依存するため、各 all-gather は pipeline を停止させず、forward または backward pass と重ねられる。

$$
\begin{aligned}
&\mathrm{Forward}(V)\to\left(\mathrm{Forward}(T)\parallel\mathrm{AllGather}(V)\right)\to\nabla_{\mathrm{Text}}\\
&\mathord{\to}\;\left(\mathrm{Backward}(T)\parallel\mathrm{AllGather}(T)\right)\to\nabla_{\mathrm{Vision}}\to\mathrm{Backward}(V),
\end{aligned}
$$

ここで $V$ と $T$ は visual feature と text feature、$(A\parallel C)$ は計算 $A$ と通信 $C$ のオーバーラップ、$\nabla$ は勾配計算を表す。この schedule では text forward pass 中に visual feature を集約し、text backward pass 中に text feature を集約するため、両方の all-gather を有用な計算の背後へ完全に隠せる。

**End-to-End Parallelism。** vision encoder と LLM の model・data heterogeneity [Zha25ax] を扱うため、近年の学習 system [Lon25, Kim26b] で用いられる disaggregated encoder 設計を採用する。vision encoder を LLM parameter tree の外に複製し、各 training step を vision encoder forward、LLM forward/backward、vision encoder backward の 3 段階に分ける。この分離により vision encoder と LLM の計算が干渉しない。load-balanced vision processing は最初と最後の段階だけで行い、LLM phase には vision computation を含めず text-only training の parallel strategy を保つ。

**長系列マルチモーダル学習の最適化。** DeepSeek-V4.1-Flash は最大 100 万トークンの系列で学習し、事前学習と事後学習の両方で相当部分を超長系列が占める。この系列長では、マルチモーダルサンプルが I/O、CPU、メモリに大きなボトルネックを生む。

- **均衡画像シャーディング。** 事前学習時、画像密度の高い単一の超長系列が読み込み中に 1 host の I/O、CPU、メモリを使い切る可能性がある。そのため各系列の画像を load balancing しながら CP rank 間で shard し、各画像をちょうど 1 回だけ読み込む。画像を一度だけ読む場合、次が成り立てば読み込みは常に計算の背後に隠れる。

$$
\frac{N\times\rho}{B_{\mathrm{IO}}}<\frac{N\times C}{B_{\mathrm{GPU}}}\Longleftrightarrow\rho<\frac{B_{\mathrm{IO}}}{B_{\mathrm{GPU}}}C,
$$

ここで $N$ は token count、$\rho$ は token 当たりの raw byte、$C$ は token 当たりの compute、$B_{\mathrm{IO}}$ と $B_{\mathrm{GPU}}$ は file system と GPU の bandwidth である。$N$ は相殺されるため、条件は token 当たりの量（$\rho$ と $C$）だけに依存し、系列長や cluster size には依存しない。$\rho$ は resolution cap や spatial downsample など vision module configuration で決まる。したがって storage throughput がボトルネックになるのは、ablation のように token 当たり compute が少ない小規模モデルだけであり、本番規模のモデルは compute-bound のままである。

- **増分画像転送。** 上記の均衡シャーディングに加え、reinforcement-learning rollout では画像を inference engine へ増分的にのみ転送し、engine の CPU-side decoding・preprocessing output を distributed file system に cache して、rollout と後続学習で再利用する。

<span id="section-3-1-2"></span>

#### 3.1.2 CSA2 のアテンション共有学習

[第 2.3 節](#section-2-3)では、main KV、indexer K、Top-K インデックスのうち 1 つ以上を複数層で共有する 3 モードの attention sharing method CSA2 を導入した。大規模分散学習で CSA2 を支えるには、attention computation 以外にも調整が必要になる。とりわけ attention component を共有する層が異なる pipeline stage に配置され得るため、従来の stage-local execution では module の直接再利用と両立しない。そこで CSA2 学習を支えるため、複数の設計を採用する。

shadow indexer は、共有 parameter に単一の logical owner を残しつつ、参加する各 stage に実行可能な軽量 replica を置くことでこの問題に対処する。owner は optimization と checkpointing を引き続き担当し、parameter synchronization と gradient aggregation によって学習中の shadow replica を一致させる。これにより、pipeline scheduler が共有層を特殊な execution unit として扱わなくても、元の model semantics を保てる。

pipeline payload extension は、source layer と consumer layer が pipeline boundary をまたぐ場合に、downstream consumer が必要とする intermediate representation と sparse routing information を渡す。これらの state を既存の point-to-point communication path に組み込み、context parallelism と整合する形で partition することで、対応する gradient flow を保ちながら不要な複製を避ける。

micro-batch-level shared-state management は、同時に active な pipeline micro-batch に紐づく state を追跡し、forward execution、activation recomputation、backward propagation にわたって lifetime を調整する。shared state は最後の consumer が完了するまで保持し、その後すぐに解放して追加 memory overhead を抑える。同じ runtime abstraction が stage placement と source-consumer relation も扱うため、attention implementation は物理的な pipeline layout に依存せず shared state にアクセスできる。

optimizer、checkpointing、warm-up、computation-graph tracing の workflow に対する軽量な調整と合わせ、これらの仕組みにより CSA2 は既存の distributed training interface と pipeline schedule のもとで透過的に動作する。

<span id="section-3-1-3"></span>

#### 3.1.3 Engram

Engram embedding table は、engram parallel size 専用 process group 間で row ごとに partition する。group size は device 当たり memory usage と embedding lookup の communication scope のトレードオフを制御する。optimizer state は各 table partition の replica 間でさらに shard する。Engram lookup index は input token sequence だけで決まる。そのため各 pipeline stage が現在の training step の microbatch を処理し始める前に、local batch 全体の embedding prefetch を開始し、pipeline execution execution への干渉を最小化する。embedding gradient は backward 中に buffer し、backbone backward pass の後で owner rank に返す。マルチモーダル学習へ効率よく統合するため、embedding prefetch と gradient transfer を vision encoder の forward・backward computation と重ねる。embedding は FP8 で保存・取得し、取得値と scaling factor を後続 GEMM へ直接渡す。Engram table update では、Sinkhorn normalization が反復間で row・column scaling vector を維持し、正規化済み matrix 全体の書き込みを繰り返さずに済むようにする。row normalization と partial column statistic の accumulation を単一 kernel に融合し、memory traffic をさらに減らす。RL rollout 中は Engram embedding table を GPU memory に常駐させる。この配置は host memory pressure を軽減し、host memory fragmentation による out-of-memory failure を防ぎやすくする。

<span id="section-3-2"></span>

### 3.2 推論システム

DeepSeek-V4.1-Flash は推論効率を第一級の要件として設計した。アーキテクチャは概念上複雑だが、得られる inference kernel flow は驚くほど簡潔である。適切な kernel fusion により複雑な演算を包み込み、少数の fused kernel 内で hardware resource を完全に pipeline 化する。これには FlashMLA [Fla25] の fused-RoPE-attention-RoPE-cast kernel、DeepGEMM [Zha25f] の Mega-Gate、Mega-mHC、Mega-MoE kernel、TileKernels [Wan26a] の kernel、DeepSelect [Qia26] の TopK kernel が含まれる。その結果、大多数の Transformer layer、すなわち CSA2 が Reuse Mode で動く層は、プリフィル時 15 kernel、decode 時 11 kernel だけで実行され、高 throughput と低 latency を両立する。

デプロイレベルでは Encoder-Prefill-Decode（EPD）分離を採用し、vision encoding、prefill、decoding を独立に scale させ、実行を overlap できるようにする。

<span id="section-3-2-1"></span>

#### 3.2.1 永続 KV キャッシュ管理

同一ワークロードで、V4.1 は永続 KV キャッシュの占有量を V4 の 8 分の 1 に減らす。2 つの乗算的要因がこの削減をもたらす。永続 KV キャッシュに SWA KV を保存しなくなってサイズがほぼ半減し、残る global KV もアーキテクチャと精度の最適化によって V4 の 4 分の 1 へ圧縮される。

V4 のデプロイでは、SWA KV が永続 KV キャッシュ容量のほぼ半分を占める。このキャッシュ内で global KV と SWA KV は独立に管理し、LRU eviction policy に従う。global KV は全体を保存し、hit 時には prefix 全体を再利用する。一方、SWA KV は prompt 末尾と output 末尾の 2 箇所で cache し、再生成と multi-turn session に備える。hit すれば、その位置から計算を再開できる。高い hit rate を維持するため、SSD 上の永続 KV キャッシュを十分大きく構成し、典型的なワークロードでは両種の KV が 72 時間以上常駐するようにした。指定位置に $n_{\mathrm{win}}$ 個の KV entry しか保持しないとはいえ、非圧縮 SWA KV cache は、短い turn を重ねる multi-turn conversation で特に大きな保存 overhead になる。

SWA KV の永続保存はコストが高いうえ効果も薄い。その access pattern が永続 KV キャッシュの長期保持 policy と合わないからである。long-tail reuse を示す global KV と異なり、SWA KV が再利用されるのは active session 内の数分という狭い窓に限られ、session 終了または次 turn 開始とともに不要になる。V4 technical report は Zero SWA Caching を提案し、欠落 SWA KV を再計算して保存 overhead を避けた。しかし厳密な復元には $L\times n_{\mathrm{win}}$ token の完全な forward pass が必要で、本番デプロイでは高すぎることが分かった。

そこで V4.1 は永続 KV キャッシュ管理を次のように改める。

1. SWA KV は永続 KV キャッシュに保存せず、各 machine の host DRAM の 10% から用意した distributed memory pool に置く。この pool の総容量ははるかに小さいが、TTL が数分と短いため、期限切れ entry をすぐ新規 session に再利用できる。実ワークロードでは、この高い turnover で同時進行中の active session の大半を十分処理できる。global KV は永続 KV キャッシュに残し、少なくとも 72 時間の lifetime を保証する。

2. SWA KV を eviction すれば必然的に miss が生じるが、軽量 fallback の Encoder SWA Bounded Replay（詳細は[第 3.2.2 節](#section-3-2-2)）により低コストに保てる。global KV は hit したものの SWA KV が miss する、避けられないが頻度の低い request では、完全な $L\times n_{\mathrm{win}}$-token forward pass ではなく $n_{\mathrm{win}}$ token だけを再計算して欠落 state を復元する。この bounded replay が設計の要であり、致命的な miss を緩やかで安価な劣化へ変えることで、永続 KV キャッシュから SWA KV を除く根拠となる。

<span id="section-3-2-2"></span>

#### 3.2.2 SWA Bounded Replay

SWA の依存関係は層をまたいで蓄積するため、$L$ 層の SWA KV を厳密に再構築するには $L\times n_{\mathrm{win}}$ token の replay が必要になる。SWA Bounded Replay は代わりに直近の $n_{\mathrm{win}}$ token だけを replay し、SWA を replay segment に切り詰めて近似 state を受け入れる。position $s$ から replay を始める場合、position $i$ の query は $[\max(s,i-W+1),i]$ の SWA key に attend する。

**Encoder SWA Bounded Replay。** Encoder SWA Bounded Replay は prefix caching を global KV のみに依存させ、永続 KV キャッシュから SWA KV を除けるようにする。

encoder SWA KV が欠落している場合、cached prefix 末尾の $n_{\mathrm{win}}$ token を replay し、uncached suffix と一緒に処理する。replay token は cached global KV を再計算も上書きもせず再利用し、SWA KV だけを再生成する。一方、uncached suffix は global KV と SWA KV の両方を生成する。

設計上、replay した prefix state は近似である。そのため uncached suffix について計算した global KV と SWA KV は cache-hit position に依存し、position 間で数学的に同一ではない。幸い、実験ではこの bounded replay strategy が応答品質をほとんど損なわないことを確認した。

**Decoder SWA Bounded Replay。** Decoder SWA Bounded Replay は decoder forward pass を $n_{\mathrm{win}}$ token に制限し、prefill computation 全体をほぼ半減させる。

CED では decoder global KV を encoder 最終 hidden state から射影する。encoder で prefill を終えるうえで唯一の障害は decoder SWA KV である。これは decoder 各層の hidden state から生成され、最初の decode step に必要となる。decoder SWA KV は一切 cache しないため、厳密な再構築には $L/2$ decoder layer を prompt 末尾の $L/2\times n_{\mathrm{win}}$ token に対して実行する必要があり、長い cached prefix の後に短い uncached suffix が続く場合は高コストになる。そこでこの場面にも bounded replay strategy を適用する。prefill のたびに prompt 末尾の $n_{\mathrm{win}}$ token を replay し、その encoder output を同じ SWA truncation のもとで decoder layer に通す。得られた decoder SWA KV は decode にのみ使い、prefix caching には使わない。

設計上、再構築した decoder SWA KV は decoder の完全な forward pass から得られるものと数学的に等価ではない。しかし、この戦略が応答品質に与える影響も無視できるほど小さい。安全性をさらに高めるため、事後学習でも同じ replay を模擬し、学習時から適応させる。

<span id="section-4"></span>

## 4 事前学習

<span id="section-4-1"></span>

### 4.1 データ構築

**テキストデータのキュレーション。** より高い知能を目指し、小規模データ実験で測れるサンプル単位の一般的品質にとどまらず、固有の情報利得を持つ多様なコーパス間の全体的相互作用を重視する。より体系的で標準化したデータ構築 pipeline を採用し、データ品質を高めて混合比を最適化する。具体的には、より包括的な評価に基づき model parameter と training data の scaling ladder を綿密に設計し、大規模 training run を導く。能力の低いモデルの出力や低品質な機械翻訳文など、情報利得の小さい model-generated content を除外する。この種の content は既存情報をほぼ言い換えただけで、長い training horizon では害になり得るため、暗黙の重複とみなす。将来の大規模 synthetic data に向けた基盤として、model-in-the-loop の data iteration も探究する。さらに領域専門家を増やし、細粒度の data quality evaluation dimension を構築する。前版と比べ、新コーパスは新たに公開された open-source repository、commit、library、emerging framework から新しい code を多く取り込み、より幅広い programming language を覆い、現在の実世界 software engineering scenario をよく反映する。

**マルチモーダルデータのキュレーション。** マルチモーダル事前学習 dataset は主に image-text pair、interleaved image-text data、domain-specific data の 3 種類からなる。生の Web data はもともと豊富な multimodal knowledge を持つという前提に立ち、大規模 data synthesis は行わず、データを本来の形のまま clean して活用することを優先し、事前学習中に最も直接的かつ scalable な visual knowledge compression を実現した。初期収集では crawler が text-centric web content に偏りすぎていると分かり、Common Crawl から再 bootstrap して multimodal source の coverage を改善した。image-text data では Web page から画像と関連 alt text を取り出し、image-text relevance threshold で filter し、image semantics に基づいて deduplicate する。interleaved data は主に Web page と PDF から構築する。大規模 multimodal corpus の処理は通常、text-only corpus より CPU と disk storage のコストが高いため、interleaved-data construction をコストが段階的に上がる複数 stage に分けた。画像取得前に heuristic・statistical filtering、deduplication、quality model を適用して高価値 document を選ぶ。残った document を interleaved image-text sequence に組み立て、image-aware な filtering と deduplication を再度行う。最後に SmolVLM [Mar25] で image-text content を厳格に品質評価し、高品質 interleaved data を抽出する。過程で除外した document の一部は、screening と recombination により追加の image-text pair として再利用する。Web 収集 data の固有の限界を補うため、fine-grained visual perception（visual grounding や pointing など）、optical character recognition（OCR）、long-tail knowledge acquisition の能力を高める domain-specific dataset も取り入れる。さらに multimodal agentic understanding を向上させるため、大量の image-code pair と computer-use trajectory を収集する。

**データ統合と重複排除。** text-only data と multimodal data は別 pipeline で処理したため、両 source の和集合を最終 training corpus とした。重複 sample は text-only version を multimodal counterpart に置き換え、2 つの設定のうち epoch count が大きい方を使う。置換後の corpus は text-only と multimodal data の token ratio が 7:1 となる。sample を共同で prefetch・assign することで、事前学習と context extension 中の sample overlap を最小化する。超長 document は混合前に deterministically pre-split し、data shard と training step 間で training token を均等に分布させる。best-fit packing algorithm も改善し、padding rate を最大 $10^{-4}$ に抑えた。

<span id="section-4-2"></span>

### 4.2 事前学習設定

<span id="section-4-2-1"></span>

#### 4.2.1 モデル設定

Transformer layer 数を 40、hidden dimension $d$ を 5120 とする。Causal Encoder-Decoder を採用し、encoder 20 層、decoder 20 層で構成する。最初の 2 層には pure sliding window attention を使う。残る encoder 18 層は圧縮率 $m=2$ の CSA2 を使い、同一設定の 6 層 group 3 個に分ける。各 group の先頭層は Full Mode、残り 5 層は Reuse Mode で動作する。decoder 20 層は圧縮率 $m=1$ の CSA2 を使い、4 層 group 5 個に分ける。第 1 group の先頭層は Full Mode、残り 3 層は Reuse Mode である。残る 4 group は同じ構成で、先頭層が Reindex Mode、残り 3 層が Reuse Mode となる。全 CSA2 layer で indexer query head 数を 32、indexer head dimension を 128、sparse attention が選択する KV entry 数、すなわち attention top-k を 512 とする。query head 数は 64、head dimension は 512、query compression dimension は 1280 とする。Hierarchical Sparse Indexer では 8 position の block を最大 2,048 個選び、候補 position は合計最大 16,384 個となる。output projection group 数は 8、各 intermediate attention output dimension は 1024 とする。追加の sliding window attention branch は window size $n_{\mathrm{win}}$ = 128 とする。全 Transformer block に MoE layer を用い、threshold 10 で clamp する SwiGLU activation function [Ope25c] を使う。各 MoE layer は 1 shared expert と 384 routed expert からなり、各 expert の intermediate hidden dimension は 2304 である。routed expert のうち token ごとに 6 expert を活性化する。mHC の expansion factor は 4、Sinkhorn-Knopp iteration 数は 20 とする。vision encoder は 32 層、hidden dimension 1024、attention head 16、image patch size 14 とする。vision MLP projector は 2 層で hidden dimension 5120 である。この構成で DeepSeek-V4.1-Flash は 552B backbone parameter を持ち、token 当たりの活性化量は prefill 時 8B、decode 時 16B となる。

<span id="section-4-2-2"></span>

#### 4.2.2 学習設定

linear transformation の parameter には Muon optimizer [Kel24, Liu25]、全 RMSNorm module の weight とその他の non-matrix parameter には AdamW optimizer [Los17]、全 embedding と prediction head には Sinkhorn-balanced update を使う。AdamW の hyper-parameter は $\beta_1=0.9$、$\beta_2=0.95$、$\varepsilon=10^{-20}$、$\mathrm{weight\_decay}=0.1$ とする。Muon は momentum 0.95、weight decay 0.1 とし、AdamW learning rate を再利用できるよう各 update matrix の RMS を 0.18 に rescale する。Sinkhorn-balanced update は Muon と同じ momentum coefficient と learning-rate correction factor を使い、$K=11$、$\tau=10^{-3}$、$\varepsilon=10^{-20}$ とする。 [Che26b] に従い、Engram learning rate は 5 倍する。DeepSeek-V4.1-Flash を 45T token の multimodal data で不安定化なく学習した。batch size は学習全体で 100.6M token に固定する。learning rate は最初の 2,000 step で線形 warm-up し、28T token まで $2.6\times10^{-4}$ を保つ。28T から 40T token では cosine schedule で $2.6\times10^{-5}$ まで decay し、40T から 45T token はこの値を保つ。sequence length 64K の sparse attention で scratch から学習し、34T token 時点で 1M へ拡張する。auxiliary-loss-free load balancing は画像・テキスト token とも bias update speed 0.001 とし、単一 sequence 内の極端な不均衡を避けるため、loss weight 0.0001 の小さな sequence-level balance loss も残す。DeepSeek-V4 と同じく、事前学習では sample-level attention masking を使う。

**Vision encoder の学習。** DeepSeek-ViT は言語バックボーンへ統合する前に独立した学習段階を経る。pipeline は contrastive pre-training と autoregressive fine-tuning の 2 段階からなる。contrastive pre-training では、alt-text data 由来の約 47B image-text pair に対し、SigLIP [Zha23t] が導入した sigmoid contrastive loss でモデルを最適化する。この巨大 dataset から visual representation を効率よく学ぶため、縦横比を保って大きな画像を縮小し、最大入力解像度を 224 × 224 pixel に制限する。この段階で高解像度を使うと顕著な改善は得られるが、実験上、最終モデルへの寄与は小さい。後続の autoregressive stage が高解像度への外挿を専用に扱うため、contrastive pre-training 中の高解像度化は全体改善が小さい割に計算 overhead を大きく増やす。autoregressive fine-tuning stage では vision encoder を 4B MoE LLM へ接続し、image caption、alt text、chart、OCR を含む dataset の 236B token に対して next-token prediction objective で学習する。この段階は encoder の fine-grained visual feature model 能力を高めることを目的とする。そのため範囲外の画像を比例 scaling し、入力解像度を 544 × 544 から 1344 × 1344 pixel に制限する。この段階の後は LLM を破棄し、最適化済み vision encoder だけを後続の事前学習 pipeline に残す。同じ input-resolution policy を維持する。

<span id="section-4-3"></span>

### 4.3 評価

<span id="section-4-3-1"></span>

#### 4.3.1 評価ベンチマーク

DeepSeek-V4.1-Flash-Base を、その前身 DeepSeek-V4-Flash-Base、DeepSeek-V4-Pro-Base と比較する。世界知識、言語理解と推論、コーディングと数学、長文脈、マルチモーダル能力という 5 つの主要次元にわたるベンチマークを報告する。

世界知識ベンチマークには AGIEval [Zho23]、MMLU-Pro [Wan24c]、C-Eval [Hua23]、MultiLoKo [Hup25]、SimpleQA-Verified [Haa25]、SuperGPQA [Du25a] を含む。言語理解・推論には BigBench Hard（BBH） [Suz22]、BigBench Extra Hard（BBEH） [Kaz25]、DROP [Dua19]、HellaSwag [Zel19]、コーディング・数学には BigCodeBench [Zhu25a]、HumanEval [Che21]、GSM8K [Cob21]、MATH [Hen21]、MGSM [Shi23]、長文脈には LongBench-V2 [Bai25] を用いる。

マルチモーダルベンチマークには MMMU-Pro [Yue24]、DocVQA [Mat21]、CVBench [Ton24]、RefCOCO／RefCOCO+／RefCOCO-g [Kaz14, Nag16a, Mao16, Yu16] を含む。

<span id="section-4-3-2"></span>

#### 4.3.2 評価結果

[表 1](#table-01)に示す。現実の研究開発場面での能力をさらに評価するため、専用の社内 corpus でも perplexity test を行った。model API では perplexity test ができないため、主に自社の pretrained base model を対象とする。corpus には日々の開発を基に、社内文書、独自 code repository、学術資料を含む別の evaluation set を収集し、複雑な科学問題と最先端研究に対する reasoning、attribution、problem-solving を評価する。結果は[図 6](#figure-06)に示し、各 model の bits-per-byte（BPB）を報告する。値が低いほど性能が高い。

<span id="table-01"></span>

![表 1。 DeepSeek-V4-Flash-Base、DeepSeek-V4-Pro-Base、DeepSeek-V4.1-Flash-Base の比較。すべてのモデルを社内 framework で同じ設定により評価する。差が 0.3 以下の score は同水準とみなす。各 row の最高 score を太字、2 位を下線で示す。](../../papers/deepseek-v4-1-flash/table-01.png)

**表 1。** DeepSeek-V4-Flash-Base、DeepSeek-V4-Pro-Base、DeepSeek-V4.1-Flash-Base の比較。すべてのモデルを社内 framework で同じ設定により評価する。差が 0.3 以下の score は同水準とみなす。各 row の最高 score を太字、2 位を下線で示す。

<span id="figure-06"></span>

![図 6。 留保 evaluation set における DeepSeek-V4-Flash-Base、DeepSeek-V4-Pro-Base、DeepSeek-V4.1-Flash-Base の bits-per-byte（BPB）比較。DeepSeek-V4.1-Flash-Base は全 task で最小 BPB を達成し、強力な base model となる大きな潜在力を示す。](../../papers/deepseek-v4-1-flash/figure-06.png)

**図 6。** 留保 evaluation set における DeepSeek-V4-Flash-Base、DeepSeek-V4-Pro-Base、DeepSeek-V4.1-Flash-Base の bits-per-byte（BPB）比較。DeepSeek-V4.1-Flash-Base は全 task で最小 BPB を達成し、強力な base model となる大きな潜在力を示す。

<span id="section-5"></span>

## 5 事後学習

<span id="section-5-1"></span>

### 5.1 事後学習パイプライン

今回の release では、新しい事後学習 algorithm を導入しない。全体の recipe は教師あり fine-tuning（SFT）に続いて reinforcement learning（RL）と on-policy distillation（OPD; [Gu25, Lu25]）を行う標準 paradigm に従い、確立済みの手法を越える algorithmic modification はない。代わりに、最適化方法ではなく学習対象へほぼすべての労力を集中し、data synthesis と environment construction の大規模自動 pipeline に投資する。具体的には、（i）多様で検証可能な training task とその reference solution・reward signal を合成し、（ii）trajectory を低コストで収集・評価できる interactive agent environment を手続き的に構築・scale し、（iii）厳格な filtering、deduplication、difficulty calibration により data quality と curriculum balance を確保する。平凡で固定的な optimization procedure のもとでは、合成 data と environment の規模、多様性、検証可能性を体系的に改善したことが、観測された向上のほぼすべてを説明する。この観察は、現段階では data・environment pipeline の engineering から得られる限界収益が、事後学習 algorithm の新規性を大きく上回るという、より一般的な教訓と一致する。

<span id="section-5-1-1"></span>

#### 5.1.1 大規模エージェントタスク合成

task は agent learning の根本的な燃料である。しかし、高品質 training task の構築には従来多大な人手が必要だった。モデルは自身の training task を構築する能力をすでに示し始めているが、まだ完全には程遠い。この可能性を踏まえ、task construction と quality verification の能力強化に大きな労力を投じた。

各 task を（problem, environment, verification system）の triplet として形式化し、task が trivial でないことを保証する difficulty と、3 要素に重大な欠陥がないことを保証する correctness の 2 次元で品質を評価する。difficulty と correctness を reward signal とし、より良い task を構築できるようモデルを反復学習する。RL task は lifecycle 全体も監視する。task を新たな RL run で使うたび、その trajectory が品質再監査の新しい証拠となる。この一般枠組みのもと、general agent と coding agent という 2 つの主要場面に専用の training environment production pipeline を構築した。

**General agent。** general agent では、社内従業員と外部 partner に最新モデルを日常 workflow へ取り入れてもらい、任意で interaction data と feedback を返してもらう。返却 data で観測した interface に基づき、現実の tool・system の interface と behavior を再現する大量の mock tool を構築する。input format、output structure、API schema、behavioral constraint を含み、一般的な SaaS・enterprise application から専門的な business backend system までを対象とする。同時に、社内従業員から寄せられた negative feedback と model failure case を大規模収集し、pipeline に組み込んで、実 workflow に根ざす single-turn・multi-turn agent environment を生成する。関連 tool context、user interaction pattern、failure condition を再構築することで、failure を体系的に replay し、観測した model weakness に対する targeted reinforcement learning を可能にする。

**Coding agent。** coding agent の training environment は 2 source から構築する。1）社内従業員と外部 partner の coding-agent session から、高度に複雑な task または model performance が低い task を残し、trajectory で deduplicate したもの。2）star count threshold を満たす public GitHub repository。environment construction は複数の専門 agent が協働して行う。まず agent が project を container 内で build・完全実行できるか、自動検証できるかを判定する。可能なら特定の turn または commit を task starting point とし、十分複雑な implementation direction を複数設計し、fail-to-pass と pass-to-pass の point を含む具体的な evaluation point と construction report を作る。必要に応じて Web から外部 resource を取得する。次に別の agent が isolated container 内で dependency、initial working directory、test code、task description を設定して self-test し、task solution を漏らす痕跡を除き、environment を新しい image layer として package する。その後、複数の異なる agent が task を試し、独立した quality-inspection agent が environment と solving agent の trajectory を確認する。environment issue、factual error、evaluation point と task description の不一致、hackability risk を検査する。合格しなければ repair agent が指摘された全 error を修正し、容易すぎる／難しすぎる evaluation point を調整し、task を再検証する。

これらの pipeline により、正しく識別力があり、長さと難易度を制御できる RL training data を自動かつ一括で生成できる。general agent environment で現実の workflow を忠実に再構築する場合も、coding agent environment で coding task を精密に構築する場合も、最終的には統一 training system へ入り、継続的な quality monitoring のもとで model capability を反復改善する。

<span id="section-5-1-2"></span>

#### 5.1.2 合成タスク上の RL

合成 task 上の大規模 asynchronous RL により model performance を高め、複雑な場面での振る舞いを形成する。RL run を training compute と scaffold 数の 2 次元で scale する。[図 7](#figure-07)と[図 8](#figure-08)に示すように、単一 scaffold 内、同一 scaffold の variant 間共同、heterogeneous scaffold 間のいずれで scale しても、cumulative RL step とともに compute を増やすほど performance は向上し続ける。

<span id="figure-07"></span>

![図 7。 DeepSeek Harness の Minimal mode で RL training を拡大するにつれ、各種 code agent benchmark の performance が向上する。最大 context length を 1M token へ伸ばすと、Terminal-Bench v3.0 など極端に長期の task でもさらに改善する。](../../papers/deepseek-v4-1-flash/figure-07.png)

**図 7。** DeepSeek Harness の Minimal mode で RL training を拡大するにつれ、各種 code agent benchmark の performance が向上する。最大 context length を 1M token へ伸ばすと、Terminal-Bench v3.0 など極端に長期の task でもさらに改善する。

<span id="figure-08"></span>

![図 8。 Claude Code の複数 version を共同学習した場合（左）と、OpenCode、Pi、DeepSeek Harness の Standard・PTC mode を含む heterogeneous scaffold 間で共同学習した場合（右）のいずれも、cumulative RL step とともに performance が向上する。DeepSWE v1.1 で評価した。薄い curve は個別の scaffold version または scaffold の評価を示す。](../../papers/deepseek-v4-1-flash/figure-08.png)

**図 8。** Claude Code の複数 version を共同学習した場合（左）と、OpenCode、Pi、DeepSeek Harness の Standard・PTC mode を含む heterogeneous scaffold 間で共同学習した場合（右）のいずれも、cumulative RL step とともに performance が向上する。DeepSWE v1.1 で評価した。薄い curve は個別の scaffold version または scaffold の評価を示す。

多様な scaffold 間で RL training するため、agent rollout execution を agent sandbox と worker container に分離する。sandbox が scaffold とその tool を実行し、worker は scaffold-agnostic control layer として rollout を編成し、異質な interaction を共通 trajectory schema へ正規化し、trainer と通信する。両者は preemptible GPU training pool の外にある DSec（[第 5.1.3 節](#section-5-1-3)）で動作し、長寿命 rollout と細粒度 training scheduling を分離する。trainer preemption 中には rollout execution を suspend・offload し、CPU・GPU resource を解放しつつ完全な state を保存して後で再開できる。この設計により、基礎 algorithm を変えずに heterogeneous scaffold 間で安定かつ効率的な RL を行える。

単一 training run を越えて effective RL compute を広げるため、model merging で successive RL run を再初期化する。具体的には異なる scaffold または configuration の run から checkpoint を merge し、異なる optimization path で得た改善を統合する。[図 7](#figure-07)と[図 8](#figure-08)の切れた curve segment は、model reinitialization 後の successive RL run を示す。これにより task performance と token efficiency がさらに向上し、parallel RL compute を集約して後続 run へ scale し続ける簡潔で実用的な方法となる。

<span id="section-5-1-3"></span>

#### 5.1.3 エージェントを大規模実行する DSec

DeepSeek-V3 から V4 への移行に伴い、agentic training environment の数と多様性が急増したため、大規模 agentic training・evaluation 向け本番 sandbox platform DeepSeek Elastic Compute（DSec）を構築した。初期設計では heterogeneous execution environment、scalable image distribution、複数 isolation backend、high-density resource management、command trajectory logging、preemption-safe resumption に対応した。

V4.1 の学習では、多様な harness、platform、code repository、software dependency、task-specific service にまたがる数百万の concurrent sandbox instance がさらに必要になった。この規模では主な bottleneck が datacenter scalability、workload isolation、node 当たり compute density、高性能化する agent の不正動作封じ込めへ移った。以下で主な設計を簡潔に説明する。

**大規模な水平計算スケーリング。** DSec は sharding と relaxed consistency scheduling という相補的な 2 つの仕組みで scale する。大量の machine に対応するため、compute node を複数 shard、いわゆる scale unit に分ける。この sharding は異なる experiment の workload を隔離して blast radius を抑え、単一の memory-intensive task が無関係な作業と共有する resource を枯渇させるのも防ぐ。

Kubernetes のような既製 orchestrator ではなく、DSec は strong global consistency と scalability を交換する独自 placement engine で大量の sandbox を schedule する。この設計は、各 compute node が local safety constraint を強制する限り、agentic sandbox placement には eventual consistency だけで足りるという観察に基づく。placement engine は同期調整せず、複数の独立 replica として配置する。各 replica は最近の測定から resource availability を予測し、十分に妥当な placement decision を行う。consistency の低下を補うため、各 node が最終 placement decision を検証し、local warning threshold を超える場合は新規 placement を拒む hard admission constraint を強制する。全体として、central coordination が bottleneck にならずに DSec を数百万 container まで scale できる。

**高密度 sandbox 実行。** node level では hardware support 付き sub-NUMA partitioning を使い、各 worker VM を個別の NUMA domain に bind する。container はこの worker VM 内で動き、CPU と memory allocation を VM の local NUMA resource に限定する。この構成は積極的な memory overcommitment と Linux kernel lock contention の均衡を取りつつ、memory pressure と runtime failure を局所化する。同等の workload configuration では、測定可能な end-to-end degradation が現れるまでの対応密度を、physical node 当たり約 1,000 から 2,500 超の concurrent live container へ高める。

ただし、このような高密度デプロイは background workload の干渉により time-sensitive evaluation を歪め得る。そのため DSec は latency-sensitive（LS）execution class を導入する。non-LS task に SCHED_IDLE を適用して scheduling priority を最小化し、core scheduling によって同じ priority class の task だけが sibling hyperthread で同時実行されるようにして干渉をなくす。

**不正動作するエージェントの緩和。** RL training 中、agent が reward hacking を試みたり、意図せず environment を crash させたりすることが頻繁にある。XFS driver の permission issue、AppArmor の illegal memory access、package mirror service からの answer leakage など、公開直後の vulnerability を悪用した例もあった。critical binary の削除、system file の破壊、filesystem 全体の消去も起きる。sandbox ごとの AppArmor profile と細粒度 eBPF-based network policy でこれらを防ぐ。agent が environment を crash させた場合、その crash を failed trajectory とし、RL framework に「repercussion」signal を返す。

<span id="section-5-1-4"></span>

#### 5.1.4 RL における推論量の制御

model architecture と hardware の進歩に加え、output token 数も serving cost、ひいては実アプリケーションの cost-quality trade-off を決める主要因である。そこで reinforcement learning 中の明示的 conditioning signal として scalar effort level $b$ を導入する。この仕組みは single-turn reasoning と multi-turn agentic task の両方に適用する。具体的には system prompt の先頭に次の instruction を加える。

Reasoning Effort: {effort}（範囲 1-100。値が高いほど、より綿密な推論を求める）

ここで $b\in\{1,\ldots,100\}$ は要求 effort level を表す。

各 training prompt $x$ について、各 effort level $b\in\mathcal{B}$ で $M_b$ 個の response を sample する。

<span id="equation-08"></span>

$$
z_{b,j}\sim\pi_\theta(\cdot\mid x,b),\qquad b\in\mathcal{B},\qquad j=1,\ldots,M_b.
$$

ここで $j$ は effort level $b$ で sample した response を index する。同じ$(x,b)$を共有する response は subgroup を形成し、その内部で reward を mean-center して group-relative advantage を計算する。したがって異なる effort level の response を直接比較しない。代わりに reward の length component を $b$ に依存させ、subgroup 内で effort-dependent behavior を誘導する。具体的には response $z_{b,j}$ の reward に length-penalty term $r_{b,j}^{\mathrm{len}}$ を加える。

<span id="equation-09"></span>

$$
r_{b,j}^{\mathrm{len}}=-\min\left\{C_{\max},\,k(b)\frac{\ell_{b,j}}{L_{\mathrm{norm}}}\right\},
$$

ここで $\ell_{b,j}$ は reasoning token 数、$L_{\mathrm{norm}}$ は reference length、$C_{\max}$ は trajectory に適用する最大 deduction の上限である。token-penalty coefficient は要求 effort とともに指数的に減少する。

<span id="equation-10"></span>

$$
k(b)=k_0\exp\left(-\frac{b-b_{\min}}{\tau}\right),\qquad \tau=\lambda\overline{\Delta b},
$$

ここで $k_0$ は各 effort level の basic penalty coefficient、$b_{\min}$ は $\mathcal{B}$ の最小値、$\overline{\Delta b}$ は training effort level 間の平均間隔、$\lambda$ は penalty decay rate を制御する。$b$ を $\tau$ 増やすと penalty coefficient は $e^{-1}$ 倍になる。$k_0$ は短い reasoning を促す全体圧力を制御し、$\tau$ が小さいほど penalty は急速に減衰して、effort level 間の behavior separation が大きくなりやすい。付録 C では $k(b)$の exponential parameterization を marginal utility の観点から説明する。

デプロイ時、scalar $b$ は model の reasoning strength を柔軟に制御する interface となる。$b$ を変えることで、単一 model checkpoint が学習済み cost-quality frontier 上の異なる operating regime 間を移動し、latency、token budget、solution quality の各要件へ適応できる。学習で使う effort level は有限だが、デプロイ時には中間値から補間された reasoning behavior を引き出せるため、test-time resource allocation を細粒度かつ効率的に制御できる。

2026 年 9 月に開始した本番デプロイでは、public API が max、high、low の 3 種類の preset reasoning-effort tier を公開し、この scalar interface に直接対応させる。

[表 2](#table-02)に示すとおり、3 tier はそれぞれ effort value $b=100$、$b=75$、$b=50$ に対応する。API user は model weight や decoding configuration を変えず、学習済み cost-quality frontier 上の operating point を選べる。

<span id="table-02"></span>

![表 2。 public API の reasoning-effort tier と基礎となる scalar effort value $b$ の対応。](../../papers/deepseek-v4-1-flash/table-02.png)

**表 2。** public API の reasoning-effort tier と基礎となる scalar effort value $b$ の対応。

LLM の RL rollout phase における long-tail problem は、学習効率を妨げる主要な bottleneck であり続けている。これに対処するため、事後学習基盤を拡張して sample を asynchronous generation [Zen26, Kim26b] できるようにし、十分に高い concurrency を維持することで rollout phase の long-tail issue を大幅に緩和する。現在はほぼすべての RL・OPD task で asynchronous training が有効になり、rollout efficiency は大きく改善した。

<span id="section-5-2"></span>

### 5.2 非同期事後学習基盤

<span id="section-5-2-1"></span>

#### 5.2.1 全体ワークフロー

rollout と training を同じ physical device に colocate し、時分割実行することで、2 phase 間の resource allocation を手作業で調整せずに済む。各 task は in-flight sample 数の上限を指定し、system は rollout phase 全体でこの上限を維持する。

目標 rollout concurrency を保つため、3 種類の dispatch granularity を評価した。最終的には sample-level dispatch を採用した。新たに完了した sample 数が次の prompt に割り当てた GRPO group size に達すると、その完了をどの group が生成したかに関係なく prompt を dispatch する。これにより学習中の rollout concurrency を安定して保てる。この方法に決める前に、2 種類の granularity を試した。第 1 の試みでは開始時に追加 batch を複数 dispatch し、各 training iteration 後に full batch を補ったが、training metric が激しく振動し、batch-level granularity は粗すぎると分かった。第 2 の試みでは prompt-level dispatch に切り替え、1 GRPO group が完了すると新 prompt を dispatch した。しかし GRPO group 内の long-tail sample で容易に stall し、目標 concurrency を滑らかに維持できなかった。

十分な training sample が蓄積すると、training が進行中の rollout を preempt する。training 中は concatenated routing-replay を用いる。複数 checkpoint にまたがる sample では、routing information を新 checkpoint で破棄・再計算する代わりに、各 rollout segment で生成した expert routing を連結する。

<span id="section-5-2-2"></span>

#### 5.2.2 長さバイアスとオフポリシー効果の緩和

asynchronous generation は rollout efficiency を効果的に高める一方、training quality を下げ得る 2 つの副作用を生む。第 1 に、短い sequence ほど先に完了して初期 training batch を支配するため、特に学習初期に length-distribution bias が生じる。第 2 に、token の一部または全部を以前の checkpoint が生成した off-policy sample が必然的に生じる。この 2 問題には異なる対処が必要である。

length bias には 2 つの仕組みを用いる。第 1 に dispatcher は dataset ごとに concurrency を制限できる。これにより steady-state training batch に占める各 dataset の割合を調整し、incoming sample の source を制御して length skew を間接的に緩和する。第 2 に、早く返った short sample の破棄を支援し、steady-state length distribution への移行を滑らかにして、model が過度に短い sequence へ過学習するのを防ぐ。

off-policy 問題にはさらに 2 つの仕組みを実装する。第 1 に sample dispatch logic と training sample waiting condition を調整して maximum off-policy ratio を制限し、training data が current model から過度に乖離しないようにする。第 2 に training 中、staleness が過大な token の寄与を除く loss masking scheme を加え、古い sample が gradient update に及ぼす悪影響を抑える。

<span id="section-5-2-3"></span>

#### 5.2.3 性能最適化

asynchronous RL framework では、rollout phase を定期的に中断して更新済み policy checkpoint へ切り替える。この処理を seamless にし、中断をほぼ瞬時に行い、rollout が停止しなかったかのように再開できることを目指す。

rollout をすぐ止めるため token-level interruption を支援し、任意の token boundary で generation を停止できる。十分な training data が集まると、全 in-flight sample がほぼ直ちに止まり、遅延なく training phase へ移れる。

checkpoint switch をまたいで rollout progress を保つため、KV cache や expert routing などの rollout state を generation 中に token granularity で永続化する。新 checkpoint で再開するときは保存済み state を直接再利用し、re-prefill cost をなくして、中断 sample を停止位置から正確に続行する。全 in-flight sample の state を保持する必要があるため、sample-grained garbage collection を行い、各 sample が完了するとすぐ state を解放する。

同じ fast-interruption・seamless-resumption の仕組みにより、checkpoint switching だけでなく cluster scheduling の preemption signal にも progress を失わず迅速に応答でき、cluster utilization 全体が高まる。

<span id="section-5-2-4"></span>

#### 5.2.4 大規模オンポリシー蒸留

事後学習の最終段階では、40 を超える teacher model を使い、全 domain の dataset で final full-vocabulary OPD task を学習する。rollout efficiency を高めるため、ここでも asynchronous generation を採用する。domain ごとに training procedure が異なるため、最良の teacher は model development の異なる stage に由来し得る。また teacher model 同士や student とで architecture が異なる場合もある。事後学習基盤はこの設定に容易に対応し、実質無制限の architecturally heterogeneous teacher による full-vocabulary OPD と、無視できるコストでの効率的な切り替えを支える [Dee26]。

OPD stage は training 中の dynamic reconfiguration も必要とする。model capability を継続的に追跡し、dataset mixture、dataset ごとの concurrency limit、active teacher など training recipe を調整する場合がある。rollout batch が明確な configuration boundary となる synchronous training では変更は容易である。一方 asynchronous setting では、異なる configuration で生成した sample が in flight で共存し得る。本基盤は rollout も training も妨げず、configuration 間の一貫した移行を支える。

<span id="section-5-3"></span>

### 5.3 評価

<span id="section-5-3-1"></span>

#### 5.3.1 評価設定

事後学習の評価は主に reasoning と agentic capability に焦点を当てる。knowledge-intensive performance はほぼ事前学習で決まり、[表 1](#table-01)に報告した。

reasoning は GPQA Diamond [Rei23]、Humanity's Last Exam [Pha25]、Codeforces（internal benchmark）、MathArena Apex [Dek25] で評価し、temperature と top-$p$ は 1.0 とする。agentic capability は次の 4 category で評価する。

- **Code agent：** Terminal-Bench 2.1 [Mer26]、Terminal-Bench 3.0 [Mar26a]、Terminal-Bench 4.0 [Mar26]、DeepSWE v1.1 [Dee26c]、ProgramBench [Yan26a]、NL2Repo-Bench [Din25]。

- **Cyber security：** SEC-Bench Pro version 260505 [Lee26]、CyberGym [Wan25c]、ExploitGym [Wan26b]。

- **General agent：** AutomationBench v1.0.6 の public evaluation set [She26]、Agents' Last Exam [Sun26a]（ALE-CLI）。

- **Visual agent：** Chartography [Gar26]、BabyVision [Che26]、ZeroBench の main set [Rob25]。

code agent では、DeepSeek Harness の Minimal mode、1M-token context window、temperature 1.0、top-p 0.95 で DeepSeek-V4.1-Flash を評価する。公式設定へ合わせ、DeepSWE v1.1 には mini-SWE harness を使う。SEC-Bench Pro には session compact 設計を持つ Claude Code harness を専用に使う。visual agent task は Claude Code harness、512k-token context window、temperature 1.0、top-p 0.95 で評価する。Agents' Last Exam と AutomationBench は公式 scaffold で評価する。ほかの coding scaffold における性能は[表 4](#table-04)に示す。

coding agent evaluation の reward hacking を抑えるため、Internet access を制限し、environment から Git history を除く。さらに Go module cache（go/mod）、node modules dependency artifact、compiled JAR file、Python pycache directory など、一時的な build・package cache を多様な environment から自動除去する。それでも CyberGym の vulnerability を見つけるために Ubuntu Linux の core package を decompile するなど、exploit-seeking behavior が test 中に観測された。model の能力が高まるにつれ、Docker container や validation script など標準 evaluation infrastructure は model gaming の影響を受けやすくなる。次世代 benchmark の設計では、この振る舞いの検出と緩和を優先するよう研究 community に強く求める。[表 3](#table-03)に示すとおり、DeepSeek-V4.1-Flash は前身 DeepSeek-V4-Flash と比べ reasoning・agentic benchmark の両方で大幅に向上し、最上位の open-source・proprietary model と同等以上の性能を示す。visual agent task でも、特に visual reasoning と複雑な professional chart の分析を要する場面で堅牢な能力を示す。leading open-source model Kimi-K3 を上回る一方、leading closed-source alternative との間には測定可能な差がなお残ることを認める。

<span id="table-03"></span>

![表 3。 DeepSeek-V4.1-Flash と closed／open source model の比較。† は HLE の text-only subset を表す。最高値を太字、2 位を下線で示す。](../../papers/deepseek-v4-1-flash/table-03.png)

**表 3。** DeepSeek-V4.1-Flash と closed／open source model の比較。† は HLE の text-only subset を表す。最高値を太字、2 位を下線で示す。

<span id="section-5-3-2"></span>

#### 5.3.2 評価結果

peak performance に加え、DeepSeek-V4.1-Flash は reasoning-effort setting を公開し、inference cost と accuracy を制御可能に交換できる。[図 9](#figure-09)に示すように、reasoning・agentic benchmark のいずれでも effort level とともに accuracy と output length が着実に増える。effort を 25 から 100 へ上げると、reasoning-intensive benchmark 8 個の平均 Pass@1 は 67.1% から 76.3%、DeepSWE v1.1 は 66.0% から 74.2%、Terminal-Bench 2.1 は 82.4% から 90.6% へ向上し、output token は約 2.5 倍になる。注目すべきことに、single-response reasoning で学んだ effort control は long-horizon agentic trajectory に忠実に転移し、turn 全体の exploration・verification 量を制御する。改善は前半に集中する。60-80 の範囲ですでに最大設定の accuracy の大半を半分未満の token budget で回復する一方、最後に effort 100 へ上げると agent trajectory は 1.6-1.8 倍に伸びるが改善はわずかである。したがって maximum tier は最難関 task に限るのがよく、日常的な agentic use には moderate effort level が良好な cost-performance balance を与える。

<span id="figure-09"></span>

![図 9。 reasoning effort に応じた性能と出力長。各 panel は reasoning-effort value を 25 から 100 へ変えたときの Pass@1（実線、左軸）と response 当たり mean output token（破線、右軸）を示す。reasoning-intensive benchmark の結果は 8 benchmark（AIME 2026、Apex 2025 Shortlist、GPQA Diamond、HLE、IMO-AnswerBench、LiveCodeBench、MathArena-Apex、SimpleQA-Verified）の平均である。DeepSWE v1.1 は mini-SWE、Terminal-Bench v2.1 は DeepSeek Harness（Minimal）で評価した。](../../papers/deepseek-v4-1-flash/figure-09.png)

**図 9。** reasoning effort に応じた性能と出力長。各 panel は reasoning-effort value を 25 から 100 へ変えたときの Pass@1（実線、左軸）と response 当たり mean output token（破線、右軸）を示す。reasoning-intensive benchmark の結果は 8 benchmark（AIME 2026、Apex 2025 Shortlist、GPQA Diamond、HLE、IMO-AnswerBench、LiveCodeBench、MathArena-Apex、SimpleQA-Verified）の平均である。DeepSWE v1.1 は mini-SWE、Terminal-Bench v2.1 は DeepSeek Harness（Minimal）で評価した。

<span id="section-5-3-3"></span>

#### 5.3.3 推論量ごとの性能

[図 9](#figure-09)は、幅広い budget value に対する性能と平均応答長を示す。reasoning effort を上げるほど reasoning trace は徐々に長くなり、reasoning-intensive benchmark と software engineering task の accuracy は単調に改善する。改善幅は low-to-mid budget range に集中し、その先では逓減する。RL で使う effort level は限られるが、scalar effort により特定範囲内の response length を柔軟に補間制御できる。これにより practitioner は quality、latency、token cost を滑らかな連続体上で交換できる。latency-sensitive application は少しの accuracy degradation と引き換えに low effort で動かし、難しい task は high effort で model の full reasoning capability を引き出せる。public API service では、この scale に対応する 3 種類の preset effort level を公開する。low、high、max はそれぞれ effort value 50、75、100 に対応する。

<span id="section-5-3-4"></span>

#### 5.3.4 エージェントスキャフォールド間の性能

実際には model が単一の固定 agent framework へ展開されることは少ない。scaffold ごとに system prompt、tool definition、context management strategy、interaction protocol が異なり、特定 harness に overfit した model は別環境で大きく劣化し得る。この違いに対する頑健性を評価するため、Claude Code [Cod26]、Codex [Cod26a]、OpenCode [Ope26b]、Pi [Zec26]、mini-SWE [Yan24l]、DeepSeek Harness（DSH） [Dee26d] の Minimal・Standard・PTC mode という 6 scaffold family、8 configuration を比較する。各 scaffold で model checkpoint、decoding configuration、task set は同一に保ち、native system prompt、tool schema、turn-taking logic を含む周辺 harness だけを変える。[表 4](#table-04)は Max reasoning effort（100）における DeepSWE v1.1 と Terminal-Bench v2.1 の性能を示す。model の agentic capability は、特定 harness 固有の慣習に依存せず、prompt・tool interface の異なる scaffold family 間へよく転移する。周辺 interaction protocol と tool abstraction が変わっても性能は堅牢で、agentic behavior が単一 scaffold design と強く結び付いていないことを示す。この頑健性は、agent scaffold 間の generalization を促すよう設計した合成 training data（[第 5 節](#section-5)）の environment、tool schema、interaction format の多様性と整合する。

<span id="table-04"></span>

![表 4。 Max reasoning effort における agent scaffold 間の性能。](../../papers/deepseek-v4-1-flash/table-04.png)

**表 4。** Max reasoning effort における agent scaffold 間の性能。

<span id="section-5-3-5"></span>

#### 5.3.5 マルチエージェント

複雑な task における multi-agent collaboration を調べるため、DeepSeek Harness の Agent Team mode で DeepSeek-V4.1-Flash の予備実験を行う。

**Multi-agent harness。** DeepSeek Harness の Agent Team mode を用い、lead agent が既定で spawn_teammate により名前付きの persistent teammate を非同期作成できるようにする。

各 teammate は委譲 task を受け取り、lead history を持たない fresh mode、または lead の完了済み turn の一時 snapshot を持つ fork mode で開始する。全 agent は 1 つの repository checkout を共有し、編集は直ちに相互可視となる。

agent 間の通信には durable peer mailbox を使う。send_message で送った message は、実行中 teammate の次 step boundary で届き、idle teammate の新 turn を開始するか、inactive teammate を再開する。各 teammate の task lifecycle 全体を通じて、lead は list_agents で runtime status を監視し、wait_agent で status、mailbox、shared-task の変化を待つ。task ownership、dependency、advisory write scope は、更新時の revision check を備えた 4 個の team_task_* tool により shared task board 上で維持する。介入が必要な場合、teammate の current turn を interrupt_agent で中断できるのは lead だけである。 **interrupt_agent。** 必要な作業が完了すると、lead が統合済み変更を review・test し、最終 response を生成する。

**学習。** task performance、delegation と inter-agent communication を促す collaboration bonus、効率的な coordination を促す derived latency penalty を組み合わせた RL reward で Agent Team mode を学習する。derived latency は execution event と collaboration dependency を directed acyclic graph（DAG）として表し、固定 prefill／decode rate における token count と実測 tool-execution time から cost を割り当て、critical path length を取って計算する。これにより、serving-side batching・queuing delay への感度を抑えつつ、有用な parallelism を促して不要な sequential work と synchronization に罰を与える。

**性能。** ProgramBench [Yan26a] から、reference solution が hidden test suite で少なくとも 95% の pass rate を達成する task だけを残し、high-confidence subset を構築する。この filtering により 172 個の「golden」task が残る。さらに大幅に改善した方法論を使う FrontierSWE の大規模で難しい後継 FrontierSWE v2 [Fro26] でも評価する。現在公開中の task から GPU access が必要なものを除き、no-GPU subset を作る。両 benchmark で明示的な rollout ごとの wall-clock deadline のもと single-agent と multi-agent configuration を評価する。ここで報告する結果は予備的で、観測した最強の multi-agent configuration と利用可能な最強の single-agent baseline を比較する。ProgramBench は task 当たり最大 3 rollout、configuration 当たり計画上 516 rollout を実行する。個々の rollout で score 0.95 以上となる割合 Almost@1 を報告する。FrontierSWE v2 では Mean@5 を報告する。ProgramBench の deadline は 1-12 時間、FrontierSWE v2 は 1-20 時間で評価する。各 deadline では、その時点で得られている output から metric を計算する。

[図 10](#figure-10)に示すように、両 benchmark のすべての deadline で multi-agent configuration が single-agent を上回る。ProgramBench では multi-agent configuration の Almost@1 が 1 時間の 13.59% から 8 時間で最高 30.04% へ上がるのに対し、single-agent は 12.79% と 20.39% である。FrontierSWE v2 では multi-agent の Mean@5 が 1 時間の 13.50% から 20 時間の 32.90% へ上がり、single-agent は 10.50% から 28.20% へ上がる。

<span id="figure-10"></span>

![図 10。 ProgramBench（Almost@1）と FrontierSWE v2（Mean@5）における single-agent・multi-agent configuration の test-time compute scaling。rollout ごとの wall-clock deadline を横軸とする。](../../papers/deepseek-v4-1-flash/figure-10.png)

**図 10。** ProgramBench（Almost@1）と FrontierSWE v2（Mean@5）における single-agent・multi-agent configuration の test-time compute scaling。rollout ごとの wall-clock deadline を横軸とする。

<span id="section-6"></span>

## 6 結論、限界、今後の方向性

本研究では、最大 100 万トークンの文脈に対応するマルチモーダル Mixture-of-Experts（MoE）モデル DeepSeek-V4.1-Flash を提案した。model architecture、cache precision、deployment strategy を共同最適化し、KV キャッシュ圧縮の限界を押し広げる。Causal Encoder-Decoder（CED）により、token 当たりの活性化 parameter は decode 時 16B に対し prefill 時 8B にとどまり、入力偏重の agentic workload の cost efficiency が向上する。同じ sequence length では、Compressed Sparse Attention 2（CSA2）における cross-layer KV cache reuse と FP4 KV caching により、常時 HBM に置く global KV cache は token 当たり 890 byte、DeepSeek-V4-Flash の約 4 分の 1 となる。SWA Bounded Replay は、常時 SSD または host memory に置く persistent KV cache を DeepSeek-V4-Flash の約 8 分の 1 へさらに減らす。これらの削減で HBM・SSD capacity pressure を緩和しつつ、総合性能は DeepSeek-V4-Flash を大きく上回る。GLM-5.3 や Kimi-K3 など同時代の open-source model より parameter footprint が大幅に小さいにもかかわらず、DeepSeek-V4.1 は主要 benchmark で同等、一部 task ではそれ以上の性能を達成する。

DeepSeek-V4.1-Flash は DeepSeek-V4-Flash と比べて複数の architecture component を大幅に単純化したが、新たな変更にはまだ完全に特性を把握できていない robustness boundary も生じる。社内評価は多様な test case と boundary condition を覆い、評価設定内で model capability の体系的劣化は観測していない。しかし有限の test suite では極端な input・deployment condition をすべて覆えない。CSA2 の selection error や SWA Bounded Replay の approximate state reconstruction が、未検証の boundary case で capability degradation を起こす可能性は残る。今後も stress-testing・evaluation stack を拡張し、特に長文脈の sparse retrieval と cache-resumption boundary における SWA state reconstruction を注視する。real-world workload も監視し、潜在 failure mode と robustness boundary を体系的に特定し、極端な条件下での robustness をさらに高める。

AI model が目覚ましい performance capability を獲得するにつれ、標準 evaluation benchmark は飽和しつつある。DeepSeek-V4.1-Flash は Fable-5 や GPT-6 Astra といった最上位 model に迫る性能を示し、日常利用で非常に近い user experience を提供するが、最難関 task にはなお差が残る。benchmark score の差が小さくても、複雑で高難度な reasoning や edge case において leading closed-source system の frontier capability と同等であるとは限らない。

したがって、state-of-the-art reasoning boundary を厳密に評価できるよう evaluation protocol を継続的に更新する。model cost の削減を続けるとともに、model intelligence をさらに伸ばすには data、model capacity、RL の coordinated scaling が必要だと考える。DeepSeek-V4.1-Flash を新たな出発点として model capability の限界を探究し、大規模 data synthesis と RL scaling の主要課題へ体系的に取り組む。model-harness co-design も積極的に取り入れ、system 全体を共同進化・共同最適化する。cost reduction と capability scaling を並行して進め、高性能 agent をより利用・展開しやすくし、広範な業種・場面で AI technology を採用する障壁をさらに下げたい。

<span id="section-7"></span>

## 7 著者一覧

著者は名のアルファベット順に記載する。* はチームを離れた人物を示す。

**研究・エンジニアリング。** Anyi Xu, B. Li, Bangcai Lin, Bing Xue, BingCheng Xian, Bingzheng Xu, Bochao Wu, Bowei Zhang, Boyi Deng, C.C. Yu, Chao Jin, Chaofan Lin, Chen Dong, Chenbing Wang, Chenfan Feng, Chengda Lu*, Chenggang Zhao, Chengqi Deng, Chengyuan Zhang, Chenhao Xu, Chenqi Zhao, Chenze Shao, Chuhao Wang, Chuqi Zhang, Damai Dai, Dejian Yang, Deli Chen, Di Huang, Di Wu, Donghao Li, Erhang Li, Eric Fu, F. Zhou, Fangwei Zhou, Fangyun Lin, Fangzhou Yuan, Feiyu Xia, Fucong Dai, Guangbo Hao, Guanglin Li, Guanting Chen*, Guoai Cao, Guofan Fan, Guolai Meng, Guowei Li, Haichuan Zhang, Haiyang Ma, Haiyang Shen, Han Li, Han Yu, Han Zhang, Hangyuan Deng, Hanwei Xu, Hanxiang Xu, Hanxun Zhong, Hao Guo, Hao Jiang, Hao Li, Hao Qin, Haodong Wen, Haofen Liang, Haofeng Huang, Haohua Liu, Haoling Zhang, Haoming Luo, Haoran Yang, Haotian Xu*, Haotian Yuan, Haoting Huang, Haowen Luo, Haoyang Cai, Haoyu Chen, Haozhe Ji, Hengran Zhang, Hengrui Wang, Hengxu Wu, Honghui Ding, Hongxuan Tang, Huadong Wang, Huanqi Cao, Huazuo Gao, Hui Qu, Hui Zeng, J. Yang, J.H. Jin, J.H. Zhang, J.X. Zou, Jia Yu, Jiahui Zhou, Jiajun Chen, Jialiang Huang, Jialin Zhao, Jiamin Tang, Jian Zhou, Jianan Tong, Jianwen Li, Jiaqi Zhu, Jiarui Wang, Jiasheng Ye, Jiashi Li, Jiaxin Xu, Jiaying Ding, Jibai Lu, Jiewen Hu, Jin Yan, Jincheng Zhai, Jingchang Chen, Jingcheng Hu, Jingli Zhou, Jingsheng Xu, Jingting Xiang, Jingyan Yun, Jingyang Yuan, Jingyuan Cheng, Jinhua Zhu, Jinpeng Wang, Jinyi Chen, Jinyi Hu, Jiping Yu, Jueliang Guo, Junbo Pei, Junbo Sun, Junguang Jiang, Junjie Qiu, Junkang Zhou, Junqi Liu, Junren Li, Junxian Li, Junxiao Song, Junyi Guo, Kai Dong, Kaifeng Chen, Kaige Gao, Kang Guan, Kangdong Yuan, Ke Hong, Ke Xu, Kefan Zhao, Kexin Ji, Kexin Zhang, Kexing Zhou, Kuai Yu, Lan Zhang, Lean Wang, Lecong Zhang, Lei Wang, Letian Gao, Liang Zhao, Liansheng Xu, Lihua Guo, Lingxiao Luo, Lingyue Fu, Litao Deng, Litong Wang, Liyue Zhang, Longhao Chen, Lu Chen, Luotian Huang, Luyao Ma, Luyao Wang, M.S. Di, Max Mei, Menghao Ye, Miao Cui, Mingchuan Zhang, Minghua Zhang*, Minghui Tang, Mingjing Zhang, Mingqi Wei, Mingshu Chen, Mingxing Liu, Mingxu Zhou, Mingyu Xu, Mingyu Yang, Mingze Wang, Muyang Chen, Ni Shentu, Ning Wang, Niufang Ning, Panpan Huang, Peixin Cong, Peiyi Wang, Peiyuan Xin, Pengfei Ren, Pengfei Yan, Pengle Zhang, Qi Kang, Qi Tang, Qiancheng Wang, Qiang Li, Qihao Zhu, Qingyang Li, Qinyu Chen, Qiushi Du, Qizhou Guo, Rongxian Xu, Rui Ding, Rui Hu, Rui Tian, Rui Yu, Ruidong Zhu, Ruifan Xu, Ruihan Yang, Ruihang Xia, Ruijie Lu, Ruilin Geng, Ruipeng Hong, Ruiqi Ge, Ruisong Zhang, Ruize Sun, Ruizhe Pan, Runji Wang, Runqian Chen, Runxin Xu, Ruohong Tian, Ruomeng Shen, Ruoyu Zhang, Ryan X., S.H. Liu, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shaofei Cai, Shaoheng Nie, Shaoyuan Chen, Shengding Hu, Shengkai Lin, Shengwen Ran, Shengyu Liu, Shengyuan Jia, Shi Bai, Shi Feng, Shicheng Xu, Shichun Liu, Shiqiang Hu*, Shirong Ma, Shiyu Wang, Shiyuan Feng, Shufan Gong, Shuhan Lin, Shuiping Yu, Shunfeng Zhou, Shuo Yang, Shuomeng Wang, Shuting Guo, Shuting Pan, Shuying Yu, Sinuo Cao, Siyi Lin, Sizhe Chen, Songyang Chen, Songyang Zhou, Tao Ni, Tao Yun, Tian Jin, Tian Pei, Tian Ye, Tianle Lin, Tianran Ji*, Tianyi Cui, Tianyuan Yue, Tingting Yu, Tongrui Xiong, Wangding Zeng, Wei Liu, Wei Zhang, Weibin Xu, Weihao Zeng, Weilin Zhao, Wen Liu, Wenfeng Liang, Wenjie Pang, Wenjing Luo, Wenjing Yao*, Wenjun Gao, Wenkai Shao, Wenkai Yang, Wenli Zhang, Wenlu Wang, Wenlve Huang, Wenqian Yan, Wentao Zhang, Xi Gao, Xiang He, Xiang Li, Xiangli Li, Xiangwen Wang, Xiangying Zhang, Xiankui Wei, Xiao Bi, Xiaodong Liu, Xiaohan Wang, Xiaojian Qu, Xiaokang Chen, Xiaokang Zhang, Xiaotao Nie, Xiaoyao Zou, Xiaoyuan Li, Xicheng Guo, Xieting Chu, Xin Cheng, Xin Liu, Xin Xie, Xinbo Xu, Xingchao Liu, Xingchen Liu, Xingkai Yu, Xingyou Li, Xintong Yao, Xinyang Chen, Xinyong Jiang, Xinyu Yang, Xinyu Yang, Xu Chen, Xuanyu Wang, Xubei Zhong, Xuecheng Su, Xuejie Liu, Xuheng Lin, Xujie Fan, Xuncheng Zhao, Xuwei Fu, Y.C. Yan, Y.H. Jiang, Y.T. Wu*, Y.W. M., Y.Z. Wang, Yafei Gao, Yang Yang, Yang Zhang, Yanru Ma, Yanwen Huang, Yao Li, Yao Li, Yao Meng, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yaoyang Ye, Yehang Yin, Yexinrui Wu, Yi Qian, Yi Tao, Yi Yu, Yichao Zhang, Yichen Jiang, Yicheng Wang, Yifan Ding, Yifan Shi, Yifeng Peng, Yifeng Zhai, Yijia Wu, Yiliang Xiong, Yilun Wang, Ying He, Ying Zhou*, Yingjia Luo, Yinmin Zhong, Yiping Wang, Yisong Wang, Yixiang Zhang, Yixiao Chen, Yixuan Tan, Yixuan Wei, Yiyang Ma, Yiyao Yang, Yiyuan Liu, Yizai Cai, Yizhen Wei, Yizhi Wang, Yonglun Yang, Yongqi Zhuo, Yongqiang Guo, Yongtong Wu, Yu Wu, Yu Zhang, Yuan Bian, Yuan Cheng, Yuan Ou, Yuan Sun, Yuanfan Xu, Yuanhang Sun, Yuanhao Li, Yuchen Liu, Yuchen Yao, Yudong Han, Yuduan Wang, Yuhan Wu, Yuhao Meng, Yuheng Zou, YuKun Li, Yunchuan Wang, Yunfan Xiao, Yunfan Xiong, Yupeng Chen, Yuqian Cao, Yuqian Wang, Yuqing Chen, Yushun Zhang, Yutong Lin, Yuwei Xiao, Yuxian Gu, Yuxiang Chen, Yuxiang Huang, Yuxiang Luo, Yuxiang You, Yuxin Chen, Yuxin Xiang, Yuxuan Liu, Yuxuan Zhou, Yuyang Zhou, Yuzhe Guo, Yuzhen Huang, Yuzhuo Bai, Z.Y. Z., Zanlin Ni, Zehao Wang, Zehua Zhao, Zehui Ren, Zejun Zhao, Zhangli Sha, Zhanying Wang, Zhaochen Zhang, Zhaoshuai Du, Zhe Fu, Zhean Xu, Zhenda Xie, Zheng Liu, Zhengyan Zhang, Zhenhua Dong, Zhewen Hao, Zhibang Wang, Zhibin Gou, Zhicheng Ma, Zhihao Li, Zhihong Shao, Zhihuan Huang, Zhijie Li, Zhirui Lu, Zhixian Huang, Zhixuan Chen, Zhixuan Chen, Zhixuan Pan, Zhiyu Wu, Zhizhou Ren, Zhu He, Zhuoshu Li, Zhuping Zhang, Zian Xu, Zihao Wang, Zihui Gu, Zijia Zhu, Zili Zhang, Zilin Li, Zilong Hou, Zilong Lyu, Ziqiao Wang, Ziwei Xie, Ziya Zhang, Ziyi Gao, Zizheng Pan, Zonglin Li, Zongqing Yao, Zui Chen, Zuofan Wu

**ビジネス・コンプライアンス。** Chenchen Ling, Chengyu Hou, Chong Chen, D. Li, Di Qi, Dongjie Ji, Fang Wei, Fanyi Xia, Fei Xie, Feiyi Tan, Hailong Guo, Haiyan Zhai, Hui Zhou, Huihui Tan, Huijie Li, Jia Luo, Jia Song, Jialu Cai, Jian Liang, Jiangting Zhou, Jiaqi Gao, Jiayi Shao, Jie Chen, Jieyu Yang, Jin Chen, Jingde Zhang, Jingzi Zhou, Jinqian Wang, Jinyang Liu, JinZhao Sun, Junhua Ling, Junmin Zheng, Kaicheng Yang, Ke Xu, Le Su, Leyi Xia, Liangfeng Ding, Lin Zhuo, Linwang Ma, Linyan Zhu, Liyu Cai, Luqi Yao, M.K. Zhang, Meng Li, Miao Lin, Miaojun Wang, Min Zhang, Mingming Li, Mingming Wang, Mingze Yin, Minmin Han, Nan Cao, Ning Wang, Ningxin Ma, Panpan Wang, Peihan Lin, Peng Sun, Peng Zhang, Qian Ying, Qiang Xiang, Qiao Wang, Qingmiao Mao, Qiwei Jiang, Rongli Jin, Ruyi Chen, Sha Tao, Shangmian Sun, Shaoqing Wu, Shichao Zou, Si Lei, Tianyang Zhang, Tianyu Sun, Tingting Yin, W.L. Xiao, Wei An, Wei Li, Wei Wang, Weiwei Lin, Wenqing Hou, X. Lin, Xiangfei Meng, Xianzhu Huang, Xiao Peng, Xiaoqian Li, Xiaoting Zhang, Xiaowen Sun, Xiaoxiang Wang, Xiaoyu Ye, Xinrou Zhang, Xinyu Zhang, Xue Cao, Xueyin Chen, Yanan Zhou, Yanhong Xu, Yao Xia, Yao Xu, Yi Shao, Yihong Zhang, Yiling Ma, Ying Tang, Yining Lou, Yiru Chen, Yishi Piao, Yixuan Chen, Yong Xiong, Yuchen Xuan, Yuehan Yang, Yuer Xu, Yukun Zha, Yunxian Ma, Yuping Lin, Yuting Yan, Yutong Xie, Yuwen Sheng, Yuxuan Zhu, Zekai Zhang, Zhe Ju, Zhenzhen Lin, Zheren Gao, Zheyang Sun, Zhigang Yan, Zhongyu Wu, Zi Wang, Zihua Qu, Ziling Yan, Ziyi Wan

<span id="section-8"></span>

## 8 評価の詳細

<span id="section-8-1"></span>

### 8.1 スキャフォールド設定

すべての scaffold は、[表 4](#table-04)の共通 evaluation setting を用いる Linux task container 上で動作する。各 run は benchmark task description から開始し、scaffold の runtime または evaluation integration が提供する prompt、task template、tool definition を使う。実験用 system prompt は追加しない。

<span id="table-05"></span>

![表 5。 Max reasoning effort における Claude Code version 間の性能。](../../papers/deepseek-v4-1-flash/table-05.png)

**表 5。** Max reasoning effort における Claude Code version 間の性能。

- **Claude Code（v2.1.105、v2.1.238、v2.1.251、v2.1.259）。** 各 version の native tool interface と Claude Agent SDK を使う。[表 4](#table-04)は v2.1.251、[表 5](#table-05)は 4 version すべてを比較する。
- **Codex（v0.147.0）。** adapted tool schema を備えた standard app-server mode を使う。
- **OpenCode（v1.18.15）。** shell・file tool と native task delegation を備えた build agent を使う。
- **Pi（v0.84.2）。** file・shell tool に `search`、`open_page`、`find_in_page` を公開する search extension を加え、RPC mode で使う。
- **mini-SWE。** `mini_swe_v2` port [+1] を single bash tool とともに使い、各 turn で tool call を必須とし、submission marker で終了する。
- **DeepSeek Harness（DSH）。** Minimal は single bash tool、Standard は Web search・fetch を含む initial function tool 26 個の full `sdk` profile、PTC は基礎 tool 24 個を使う TypeScript program 向け `run_code` を用いる。Standard と PTC は v0.1.1+custom.202609011522 を使う。

[+1]: mini-swe-agent、commit `04d809ceab9d`。

<span id="section-8-2"></span>

### 8.2 スキャフォールド間の推論量

[図 11](#figure-11)の 6 panel すべてで、reasoning-effort setting を上げるほど trajectory は長くなり、trajectory 当たり mean output token は全 scaffold-benchmark pair で effort とともに単調増加する。Pass@1 はこの増加と緩やかにしか連動しない。全体として改善するが応答は単調でなく、大半の panel では中間設定に plateau や dip がある。3 scaffold の calibration も異なる。DeepSWE v1.1 では Claude Code の curve が最も平坦で、追加 token も比較的少ない。DeepSeek Harness（Minimal）は最低から始まり、最大の token cost で最大の向上を得る。mini-SWE はその中間である。Terminal-Bench v2.1 では 3 scaffold が狭い範囲に収まり、maximum effort で DeepSeek Harness（Minimal）、mini-SWE、Claude Code の順になる。task が飽和に近づくと、scaffold choice は effort tier と同じかそれ以上に重要となる。

<span id="figure-11"></span>

![図 11。 reasoning effort は trajectory length を一貫して伸ばすが、coding scaffold 間の accuracy との相関は弱い。各 panel は DeepSWE v1.1（上段）と Terminal-Bench v2.1（下段）について、Claude Code、DeepSeek Harness（Minimal）、mini-SWE の 3 agent scaffold の reasoning-effort setting に対する Pass@1（%、実線、左軸）と trajectory 当たり mean output token（k、破線、右軸）を示す。全 panel は同じ checkpoint による。](../../papers/deepseek-v4-1-flash/figure-11.png)

**図 11。** reasoning effort は trajectory length を一貫して伸ばすが、coding scaffold 間の accuracy との相関は弱い。各 panel は DeepSWE v1.1（上段）と Terminal-Bench v2.1（下段）について、Claude Code、DeepSeek Harness（Minimal）、mini-SWE の 3 agent scaffold の reasoning-effort setting に対する Pass@1（%、実線、左軸）と trajectory 当たり mean output token（k、破線、右軸）を示す。全 panel は同じ checkpoint による。

<span id="section-8-3"></span>

### 8.3 推論量ごとの推論ベンチマーク詳細結果

[図 12](#figure-12)は、reasoning-effort setting が competition mathematics、science QA、open-domain knowledge、code にわたる 8 benchmark のすべてで、output length と accuracy を滑らかで整然と制御することを示す。length 面では effort を 25 から 100 へ上げると、全 benchmark の average response が予測可能に増える。2.0-3.1 倍という一様な増加で、AIME 2026 は response 当たり 4.6k から 11.4k token、MathArena Apex 2025 は 29.1k から 86.1k となる。暴走的増大も異常もなく、各 tier の compute cost を事前に見積もれる。accuracy も同じ滑らかな軌跡をたどり、すべての benchmark で望ましい方向へ反応し、effort を上げても劣化する benchmark はない。MathArena Apex 2025 は +40.3 point（25.3%→65.6%）、Apex 2025 Shortlist は +11.5 となる。すでに飽和した benchmark も安定し（GPQA Diamond +1.3、LiveCodeBench +2.6）、AIME 2026 は 100% に達する。このように model は単一の信頼できる knob を提供し、cost-accuracy operating point を制御可能かつ予測可能に動かす。各 deployment は accuracy を犠牲にせず、latency・compute budget に合う effort tier を選べる。

<span id="figure-12"></span>

![図 12。 reasoning-intensive benchmark 8 個における reasoning effort に応じた性能と出力長。各 panel は reasoning-effort value を 25 から 100 へ変えたときの Pass@1（実線、左軸）と response 当たり mean output token（破線、右軸）を示す。](../../papers/deepseek-v4-1-flash/figure-12.png)

**図 12。** reasoning-intensive benchmark 8 個における reasoning effort に応じた性能と出力長。各 panel は reasoning-effort value を 25 から 100 へ変えたときの Pass@1（実線、左軸）と response 当たり mean output token（破線、右軸）を示す。

<span id="section-9"></span>

## 9 推論量制御における指数的トークンペナルティ

scalar effort variable は hard token budget を課すことなく、deployment 時に cost-quality trade-off を制御できる。reinforcement learning 中、低 effort level には強い token penalty を適用し、高 effort level ではより多くの計算を許す。本節では exponential penalty schedule の簡略な動機を示す。

effort level $b$ で $\ell$ 個の reasoning token を生成した trajectory に対し、length deduction は次式となる。

<span id="equation-11"></span>

$$
r^{\mathrm{len}}(\ell,b)=-\min\left\{C_{\max},\,k(b)\frac{\ell}{L_{\mathrm{norm}}}\right\},
$$

ここで $L_{\mathrm{norm}}$ は reference length、$C_{\max}$ は deduction の上限である。effort-dependent token-penalty coefficient は次式で表される。

<span id="equation-12"></span>

$$
k(b)=k_0\exp\left(-\frac{b-b_{\min}}{\tau}\right),
$$

ここで $k_0$ は最低 effort level $b_{\min}$ での penalty coefficient、$\tau$ は penalty decay rate を制御する。

この選択の動機として、固定 problem $x$ を考える。$p_x(\ell)$ を、reasoning token を $\ell$ 個使った後に問題を解ける確率とする。cap が効かない領域で、望ましい reasoning length $\ell_x^*(b)$ を次のように定義する。

<span id="equation-13"></span>

$$
\ell_x^*(b)\in\arg\max_{\ell\geq0}\left[p_x(\ell)-k(b)\frac{\ell}{L_{\mathrm{norm}}}\right].
$$

interior optimum では first-order condition は次式となる。

<span id="equation-14"></span>

$$
p_x'\left(\ell_x^*(b)\right)=\frac{k(b)}{L_{\mathrm{norm}}},
$$

ここで $p_x'(\ell)=dp_x(\ell)/d\ell$ は、reasoning を追加することで solve probability が増える限界量である。

この marginal benefit が対象 operating range で近似的に指数減衰すると仮定する。

<span id="equation-15"></span>

$$
p_x'(\ell)\approx a_x\exp\left(-\frac{\ell}{s_x}\right),
$$

ここで $a_x>0$ は instance-dependent scale、$s_x>0$ は decay rate を決める。[式 12](#equation-12)と[式 15](#equation-15)を[式 14](#equation-14)へ代入すると、次式を得る。

<span id="equation-16"></span>

$$
\ell_x^*(b)\approx C_x-s_x\log k_0+\frac{s_x}{\tau}(b-b_{\min}),
$$

ここで $C_x=s_x\log(a_xL_{\mathrm{norm}})$ は $b$ に依存しない。したがってこの local model では、exponential penalty schedule により、要求 effort と望ましい reasoning length の間に単純な first-order affine trend が生じる。

2 つの effort level $b_2>b_1$ に対し、予測 length difference は次式となる。

<span id="equation-17"></span>

$$
\ell_x^*(b_2)-\ell_x^*(b_1)\approx\frac{s_x}{\tau}(b_2-b_1).
$$

したがって $k_0$ は主として短い reasoning を促す全体圧力を制御し、$\tau$ は effort に対する予測感度を制御する。

この導出は local reward-level approximation であり、測定した average length が線形または pointwise monotonic でなければならないという主張ではない。effort instruction が reasoning strategy を直接変え得ること、generation が stochastic であること、agent trajectory ごとに turn 数が異なること、subgroup reward normalization により optimization strength が変わることから、実現 behavior は外れ得る。また、分析は penalty cap が active でない interior solution を仮定する。cap に達すると marginal token penalty は 0 になり、capped region を別に考える必要がある。
