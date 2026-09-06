---
title: 'Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads'
createTime: 2026/09/06 22:18:29
permalink: /ja/papers/medusa/
pageClass: paper-reading medusa-paper
---

> [Tianle Cai](https://ctlllll.github.io/) [+author-equal] [+author-corresponding]、[Yuhong Li](https://leeyeehoo.github.io/) [+author-equal] [+author-corresponding]、[Zhengyang Geng](https://gsunshine.github.io/)、[Hongwu Peng](https://harveyp123.github.io/)、[Jason D. Lee](https://jasondlee88.github.io/)、[Deming Chen](https://ece.illinois.edu/about/directory/faculty/dchen)、[Tri Dao](https://tridao.me/)。2024 年 1 月 19 日に arXiv へ初回投稿。現行版は 2024 年 6 月 14 日改訂の v3。*Proceedings of the 41st International Conference on Machine Learning*、PMLR 235:5209-5235、2024 年 7 月に掲載。[Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774)。<a href="/paper/medusa.pdf" target="_blank" rel="noopener noreferrer">原論文 PDF</a>。[PMLR](https://proceedings.mlr.press/v235/cai24b.html)。[DOI](https://doi.org/10.48550/arXiv.2401.10774)。[TeX ソース](https://export.arxiv.org/e-print/2401.10774v3)。厳密な印刷レイアウトと参考文献については原論文 PDF を正とする。

## 概要

大規模言語モデル（LLM）の自己回帰デコードは逐次計算を必要とし、各ステップが直前の出力に依存する。さらに各ステップで全モデルパラメータを高帯域幅メモリ（HBM）からアクセラレータのキャッシュへ転送するため、ここがボトルネックとなる。投機的デコードなども提案されているが、別個のドラフトモデルの入手と保守が実装を難しくしている。本稿では、複数の後続トークンを並列予測する追加デコードヘッドにより LLM 推論を高速化する Medusa を提案する。Medusa は*木構造注意機構*で複数の候補続きを構成し、各デコードステップで同時に検証する。並列処理により必要なデコードステップ数を大幅に削減できる。用途に応じて二段階の微調整手順を示す。**Medusa-1** は*凍結した*バックボーン LLM 上で Medusa のみを微調整し、損失のない推論高速化を実現する。**Medusa-2** はバックボーンとともに微調整し、ヘッドの予測精度と高速化率を高める一方、モデル能力を保つ専用の学習手順を要する。学習データがない場合の*自己蒸留*、生成品質を維持しつつ採択率を上げる*典型採択方式*も提案する。各種モデルでの実験では、Medusa-1 は品質を損なわず $2.2\times$ 超、Medusa-2 は 2.3-$2.8\times$ の高速化を達成した。

<span id="section-1"></span>

## 1 はじめに

近年の LLM は、規模を数十億パラメータまで増やすほど言語生成品質が大きく向上する [Bro20, Cho22b, Zha22, Hof22, Ope23, Pal23, Tou23a]。しかし、同時に*推論遅延*も増え、実用上の大きな課題となる。システム面では LLM 推論は主にメモリ帯域幅に制約され [Sha19, Kim23]、算術演算よりアクセラレータの帯域幅が遅延を決める。自己回帰デコードは逐次的で、各順伝播のたびに全モデルパラメータを HBM からキャッシュへ転送しながら一つのトークンしか生成しない。このため現代のアクセラレータの演算能力を活用し切れない。

解決策の一つは、デコードの*算術強度*（総浮動小数点演算量と総データ移動量の比）を高め、*デコードステップ数を減らす*ことである。この考えに沿って投機的デコードが提案された [Lev23, Che23, Xia23c, Mia23b]。小さなドラフトモデルでトークン列を生成し、大きな原モデルが採択可能な続きを選別する。ただし適切なドラフトモデルの取得は難しく、分散システムへの統合はさらに難しい [Che23]。

本稿では別のドラフトモデルを逐次実行する代わりに、バックボーン上の複数デコードヘッドで推論を速める発想 [Ste18] を再検討し改良する。適切に用いれば投機的デコードの課題を避け、既存 LLM システムへ容易に組み込める。提案する Medusa は追加ヘッドで複数トークンを同時予測する。ヘッドは*パラメータ効率よく*微調整でき、任意の既存モデルへ追加可能である。ドラフトモデルを必要とせず、分散環境を含む既存システムへ統合しやすい。

Medusa にはさらに二つの着想がある。第一に、一ステップで候補続きを一つだけ作るのは計算資源を浪費するため、複数候補を作り、注意マスクの簡単な変更で同時検証する。第二に、投機的デコードの棄却サンプリング [Lev23, Che23] は原モデルと同分布の応答を生成できるが、高速化率をさらに上げられない。そこで Medusa ヘッドの出力から*妥当な*候補を選ぶ*典型採択*を導入する。温度をしきい値として原モデル予測からのずれを調整することで、同程度の品質を保ちながらデコードをさらに高速化できる。

予測用 Medusa ヘッドを LLM に付与するため、二つの微調整手順を提案する。計算資源が限られる場合や既存モデルの性能を変えたくない場合は Medusa-1 を用いる。必要メモリが少なく、QLoRA [Det24] のような量子化でも最適化でき、バックボーンを固定するため品質を損なわない。ただしバックボーンの能力を十分に利用しない。そこでバックボーンも微調整してヘッド精度と高速化率を高める Medusa-2 を提案する。これは資源が十分な場合やベースモデルから直接教師あり微調整（SFT）する場合に適する。鍵は、次トークン予測能力と出力品質を保ったままヘッドとバックボーンを共同学習する手順である。公開データで微調整されたモデルにはそのデータを直接使い、データがない、または RLHF [Ouy22a] を経たモデルには自己蒸留で学習データを生成する。

実験は主としてバッチサイズ 1、すなわち個人が LLM をローカル運用する状況を扱う。公開データで学習した Vicuna-7B/13B、非公開データの Vicuna-33B [Chi23a] [+1]、教師あり微調整とアラインメントを施した Zephyr-7B を評価した。Medusa は各種プロンプトで品質を損なわず 2.3〜2.8 倍高速化した。

<span id="figure-01"></span>

![Figure 1. Medusa introduces multiple decoding heads, tree-based attention, candidate verification, and acceptance in one inference pipeline.](../../papers/medusa/figure-01.png)

**図 1.** Medusa は LLM の最終隠れ状態に*複数のヘッド*を加え、後続トークンを並列予測する（[第 2.1.1 節](#section-2-1-1)）。推論時、各ヘッドは担当位置の上位予測を複数生成する。それらを候補に組み、*木構造注意*で並列処理する（[第 2.1.2 節](#section-2-1-2)）。最後に候補を検証して続きを採択する。標準的な棄却サンプリングのほか、*典型採択*（[第 2.3.1 節](#section-2-3-1)）で妥当な続きを選べる。*採択された接頭辞が最長の候補*を次のデコード段階に用いる。

<span id="section-2"></span>

## 2 手法

Medusa は投機的デコードと同じ枠組みを使い、各ステップを (1) 候補生成、(2) 候補処理、(3) 候補採択の三段階に分ける。(1) は Medusa ヘッド、(2) は木構造注意が担う。ヘッドは原モデル上にあるため、(2) の logits を次ステップの (1) に再利用できる。(3) には棄却サンプリング [Lev23, Che23] または典型採択（[第 2.3.1 節](#section-2-3-1)）を使う。全体を[図 1](#figure-01)に示す。

まず Medusa ヘッドと木構造注意を説明し、次に用途別の二段階の微調整手順を示す。最後に、学習データがない場合の自己蒸留と、デコード効率を高める典型採択を導入する。

<span id="section-2-1"></span>

### 2.1 主要要素

<span id="section-2-1-1"></span>

#### 2.1.1 Medusa ヘッド

投機的デコードでは補助ドラフトモデルが後続トークンを予測する。原モデルに採択される続きを作れるほど有効でありながら小さくなければならず、両立は難しい。既存手法 [Spe23, Mia23b] は小型モデルを別途*事前学習*することが多く、[Mia23b] では NVIDIA A100 を 275 GPU 時間使用した。別学習は分布ずれも招き得る。[Che23] は分散環境で複数モデルを提供する複雑さも指摘した。

LLM 推論高速化を簡単かつ広く利用可能にするため、機械翻訳や画像超解像で並列デコードを使う [Ste18] に着想を得る。Medusa ヘッドは原モデルの最終隠れ状態に追加するデコードヘッドである。位置 $t$ の状態 $h_t$ に $K$ 個を加え、第 $k$ ヘッドは位置 $(t+k+1)$ のトークンを予測する（原 LM ヘッドは $(t+1)$ を予測）。その語彙分布を $p_t^{(k)}$、原モデル予測を $p_t^{(0)}$ とする。[Ste18] に従い残差接続付き一層 FFN を使う。この単純な設計で十分な性能が得られ、第 $k$ ヘッドは次式となる。

$$
\begin{aligned}
p_{t}^{(k)}=\operatorname{softmax}\left(W_{2}^{(k)}\cdot\left(\operatorname{SiLU}(W_{1}^{(k)}\cdot h_{t})+h_{t}\right)\right), \\
\mathrm{where}\;W_{2}^{(k)}\in\mathbb{R}^{d\times V},W_{1}^{(k)}\in\mathbb{R}^{d\times d}.
\end{aligned}
$$

$d$ は最終隠れ層の出力次元、$V$ は語彙数である。$W_2^{(k)}$ は原 LM ヘッドと同じ値、$W_1^{(k)}$ はゼロで初期化し、初期予測を原モデルに合わせる。Llama [Tou23a] と同じ SiLU 活性化関数 [Elf17] を用いる。

Medusa ヘッドは原バックボーンと組み合わせて学習し、バックボーンを*凍結*する Medusa-1 と共同学習する Medusa-2 がある。強力な基盤モデルの表現を利用するため一 GPU でも大規模モデルを微調整でき、ヘッド分布も原モデルに揃って分布ずれを緩和する。新ヘッドは原 LM ヘッド同様の一層だけなので、提供系や分散構成を複雑化しない。学習法は[第 2.2 節](#section-2-2)で述べる。

<span id="section-2-1-2"></span>

#### 2.1.2 木構造注意

Medusa ヘッドから後続 $K+1$ トークンの確率を得て、長さ $K+1$ の候補続きを作る。投機的デコード [Lev23, Che23] は一候補をサンプルするが、複数候補なら一ステップの期待採択長を伸ばせる。一方で計算量も増えるため、木構造注意で複数候補を同時処理して均衡を取る。

<span id="figure-02"></span>

![Figure 2. We demonstrates the use of tree attention to process multiple candidates concurrently. As exemplified, the top-2 predictions from the first Medusa head and the top-3 from the second result in a total of $2\times 3=6$ candidates. Each of these candidates corresponds to a distinct branch within the tree structure. To guarantee that each token only accesses its predecessors, we devise an attention mask that exclusively permits attention flow from the current token back to its antecedent tokens. The positional indices for positional encoding are adjusted in line with this structure.](../../papers/medusa/figure-02.png)

**図 2.** 木構造注意による複数候補の同時処理。第 1 ヘッドの上位 2、第 2 ヘッドの上位 3 予測から $2\times3=6$ 候補を作り、各候補が木の枝になる。各トークンが先行トークンだけを参照する注意マスクを設計し、位置符号化の索引も合わせて調整する。

従来の因果注意と異なり、同じ続きのトークンだけを履歴とみなす。グラフ構造を注意に埋め込む [Yin21] に着想を得て、[図 2](#figure-02)のように木をマスクへ組み込む。[Mia23b, Spe23] はドラフトモデルの複数候補を統合するボトムアップ方式だが、本手法は Medusa 候補を使うトップダウン方式である。第 $k$ ヘッドの上位 $s_k$ 予測の直積で候補を作る。[図 2](#figure-02)では $s_1=2,s_2=3$ で、第 1 ヘッドの各予測に第 2 ヘッドの任意の予測が続く。仮想根を第 0 層とすれば第 $k$ 層に $s_k$ 分岐を持つ。適切なマスクと位置索引によりバッチを拡大せず同時処理でき、新規トークン総数は $\sum_{k=1}^{K}\prod_{i=1}^{k}s_i$ となる。

ここでは直積による最も単純で規則的な木を示したが、各ヘッドの上位予測の精度差を利用する高度な構成も可能である。[第 2.3.3 節](#section-2-3-3)で述べる。

<span id="section-2-2"></span>

### 2.2 学習戦略

基本形はバックボーンを凍結して Medusa ヘッドだけを微調整する。共同学習すればヘッド精度を大きく改善できるため、資源と用途に応じた二段階の戦略を提案する。

ここでは目標モデルの出力分布に合う学習データ、例えば SFT データがあると仮定する。自己蒸留で不要にする方法は[第 2.3.2 節](#section-2-3-2)で述べる。

<span id="section-2-2-1"></span>

#### 2.2.1 Medusa-1：バックボーンの凍結

凍結バックボーンでは Medusa 予測と正解の交差エントロピーを使う。位置 $t+k+1$ の正解 $y_{t+k+1}$ に対し第 $k$ ヘッドの損失は $\mathcal{L}_k=-\log p_t^{(k)}(y_{t+k+1})$ である。$k$ が大きいほど予測が不確かで損失も大きいため、重み $\lambda_k$ で各ヘッドを均衡させる。総損失は次式である。

<span id="equation-01"></span>

$$
\mathcal{L}_{\mathrm{Medusa}-1}=\sum_{k=1}^{K}-\lambda_{k}\log p_{t}^{(k)}(y_{t+k+1}).
$$

実際には $\lambda_k$ を $0.8$ などの $k$ 乗とする。バックボーンは隠れ状態だけを供給するため量子化版でメモリを減らせる。QLoRA [Det24] と同様、一枚の一般向け GPU でも大規模モデルを学習できる。例えば Vicuna 7B の 60k ShareGPT 標本は一枚の A100 PCIe で 5 時間である。

<span id="section-2-2-2"></span>

#### 2.2.2 Medusa-2：共同学習

ヘッド精度をさらに高めるにはバックボーンも共同学習する。ただし次トークン予測能力と品質を保つ専用手順が必要であり、次の三戦略を用いる。

- **結合損失**：バックボーンの交差エントロピー $\mathcal{L}_{\mathrm{LM}}=-\log p_t^{(0)}(y_{t+1})$ を Medusa 損失に加え、$\lambda_0$ で両者を均衡させる。総損失は次式である。

  <span id="equation-02"></span>

  $$
  \mathcal{L}_{\mathrm{Medusa}-2}=\mathcal{L}_{\mathrm{LM}}+\lambda_{0}\mathcal{L}_{\mathrm{Medusa}-1}.
  $$
- **学習率の分離**：十分に学習済みのバックボーンと、より多く学ぶ必要があるヘッドに別の学習率を使い、能力を保ちながらヘッドを速く収束させる。
- **ヘッドのウォームアップ**：初期の大きなヘッド損失と勾配がバックボーンを歪めないよう [Kum22] に従う二段階学習を行う。まず Medusa-1 と同様にヘッドのみ、次にバックボーンと共同学習する。単純な段階切替のほか $\lambda_0$ を漸増する方法もあり、どちらも良好に機能する。

これらによりバックボーン能力を損なわず共同学習でき、SFT と組み合わせて Medusa をネイティブ対応させられる。

<span id="section-2-2-3"></span>

#### 2.2.3 ヘッド数の選択

経験上は最大五ヘッドで十分である。五つを学習し、[第 2.3.3 節](#section-2-3-3)で最適な木構成を選ぶことを勧める。推論時に三、四ヘッドで足りれば、余分なヘッドは無コストで無視できる。

<span id="section-2-3"></span>

### 2.3 拡張

<span id="section-2-3-1"></span>

#### 2.3.1 典型採択

投機的デコード [Lev23, Che23] は原モデルと同分布の多様な出力を得るため棄却サンプリングを使うが、実装 [Gan23, Spe23] では温度上昇とともに効率が落ちる。ドラフトと原モデルが同一なら、貪欲デコードでは全出力が採択される一方、独立にサンプルする棄却法では分布が完全一致しても棄却が起きる。

実際には温度は応答の「創造性」を調整するために使われ、高温ほど原モデルがドラフト出力を受け入れる機会は多いはずである。そこで原分布との厳密な一致を求めず、もっともらしい候補を選ぶ*典型採択*を提案する。切断サンプリング [Hew22] に着想を得る（詳しくは[第 6 節](#section-6)）。*典型的*、すなわち原モデルで極端に低確率でない候補を選ぶため、原モデル確率に分布依存のしきい値を設ける。文脈 $x_1,\ldots,x_n$ と候補 $(x_{n+1},\ldots,x_{n+K+1})$ に次の条件を用いる。

$$
\begin{aligned}
p_{\text{original}}(x_{n+k}|x_{1},x_{2},\cdots,x_{n+k-1})> \\
\min\left(\epsilon,\delta\exp\left(-H(p_{\text{original}}(\cdot|x_{1},x_{2},\cdots,x_{n+k-1}))\right)\right),
\end{aligned}
$$

$H(\cdot)$ はエントロピー、$\epsilon,\delta$ はそれぞれ固定しきい値とエントロピー依存しきい値である。[Hew22] の、(1) 高確率トークンは有意義、(2) 高エントロピーなら複数の続きが妥当、という観察に基づく。各候補を評価し、条件を満たす*接頭辞*を採択する。毎回一つ以上生成するため最初のトークンは貪欲に選び*無条件*で採択し、以後に典型採択を使う。最長の採択接頭辞を最終予測とする。

温度 $0$ では最大確率トークンだけが非零となり、貪欲デコードに戻る。温度が正でも適切な $\epsilon,\delta$ なら最大確率の貪欲結果は常に採択され、高速化が最大になる。一般にも温度が高いほど採択列が長くなり、実験でも確認された。

[図 5](#figure-05)の通り、典型採択は同程度の生成品質を保ち、より高い高速化を実現する。

<span id="section-2-3-2"></span>

#### 2.3.2 自己蒸留

[第 2.2 節](#section-2-2)では目標モデルと同じ出力分布のデータを仮定したが、モデルだけが公開される場合や RLHF で分布が変わる場合もある。そこでモデル自身に Medusa 用データを生成させ、出力分布を一致させる自動自己蒸留を提案する。

類似領域の公開シードデータ、例えばチャットなら ShareGPT [Src23] のプロンプトにモデル自身で回答させる。複数ターンは順にプロンプトを与える。両役割で学習した Zephyr 7B [Tun23] などは自己対話できるため、最初のプロンプトだけで複数ターンを生成できる。

Medusa-1 には十分だが、Medusa-2 でこのデータだけを用いると品質が低下する。ヘッドなしでもバックボーン学習で劣化するため、知識蒸留 [Kim16c] と同様に正解トークンでなく原モデルの確率を教師とする。損失は次式である。

$$
\mathcal{L}_{\mathrm{LM-distill}}=\operatorname{KL}(p_{\mathrm{original},t}^{(0)}\|p_{t}^{(0)}),
$$

$p_{\mathrm{original},t}^{(0)}$ は位置 $t$ における原モデル予測分布である。

素朴には二モデルを保持してメモリが増える。そこで LoRA [Hu21] などのアダプタで微調整すれば、アダプタを切った状態が原モデルとなり、追加メモリなしに蒸留できる。これにより能力を損なわず Medusa-2 を学習できる。量子化 LoRA では教師も量子化され品質が落ち得るため、非量子化 LoRA が望ましい。

<span id="section-2-3-3"></span>

#### 2.3.3 最適な木構成の探索

[第 2.1.2 節](#section-2-1-2)の規則的な直積木は、総ノード数が固定なら最適とは限らない。各ヘッドの順位別予測精度が異なるため、その推定値を使って木を構成する。

較正データで第 $k$ ヘッドの第 $i$ 位予測精度 $a_k^{(i)}$ [+2] を測る。独立性を仮定すれば順位列 $[i_1,\ldots,i_k]$ の候補精度は $\prod_{j=1}^k a_j^{(i_j)}$ である。全組合せ集合 $I$ の各要素を木の全ノードに対応させると、採択長の期待値は次式となる。

$$
\sum_{\left[i_{1},i_{2},\cdots,i_{k}\right]\in I}\prod_{j=1}^{k}a_{j}^{(i_{j})}.
$$

ノードを一つずつ加えると、新ノードの寄与はその精度そのものである。現在の木に接続できる最高精度ノードを貪欲に加え、所望の総数まで繰り返せば期待採択長を最大化できる。詳細は[第 8 節](#section-8)。

<span id="figure-03"></span>

![Figure 3. Left: Speed comparison of baseline, Medusa-1 and Medusa-2 on Vicuna-7B/13B. Medusa-1 achieves more than $2\times$ wall-time speedup compared to the baseline implementation while Medusa-2 further improves the speedup by a significant margin. Right: Detailed speedup performance of Vicuna-7B with Medusa-2 on 8 categories from MT-Bench.](../../papers/medusa/figure-03.png)

**図 3.** 左：Vicuna-7B/13B における基準、Medusa-1、Medusa-2 の速度比較。Medusa-1 は実時間で $2\times$ 超、Medusa-2 はさらに大きく高速化する。右：Medusa-2 Vicuna-7B の MT-Bench 8 分野別結果。

<span id="section-3"></span>

## 3 実験

異なる条件で Medusa を評価する。Vicuna-7B/13B [Chi23a] で Medusa-1/2、学習手順を得られない Vicuna-33B と RLHF 済み Zephyr-7B で自己蒸留を検証する。多ターン対話ベンチマーク MT-Bench [Sto23e] を用い、詳細は[第 7 節](#section-7)に示す。

<span id="section-3-1"></span>

### 3.1 事例：Vicuna 7B/13B における Medusa-1 と Medusa-2

**実験設定。** Llama [Tou23a] から微調整した Vicuna [Chi23a] の 7B/13B/33B を使う。7B/13B は ShareGPT [Src23]、実験的な 33B は非公開データで学習された。ここでは ShareGPT で 7B/13B の Medusa ヘッドを $2$ epoch 学習する。Llama-2 から系列長 4096 で微調整した Vicuna v1.5 を用いる。

**結果。** [図 3](#figure-03)に示す。基準は Huggingface 標準実装。7B では Medusa-1 が $2.18\times$、Medusa-2 が $2.83\times$、13B ではそれぞれ $2.33\times$、$2.83\times$ 高速化した。Medusa-2 Vicuna-7B は coding で $3.29\times$、Extraction で最大の $3.62\times$ となり、モデル規模と課題をまたいで推論速度を大きく高めた。

<span id="section-3-2"></span>

### 3.2 事例：Vicuna-33B と Zephyr-7B の自己蒸留学習

**実験設定。** Vicuna-33B [Chi23a] と Zephyr-7B [Tun23] を用い、[第 2.3.2 節](#section-2-3-2)に従って ShareGPT [Src23] と UltraChat [Din23b] をシードに各約 $100k$ 標本を作る。Zephyr は一プロンプトで複数ターンを続けられる。Vicuna-33B にはシード対話の各プロンプトを順次与え、温度 0.3 でランダムサンプルする。系列長 $2048$、バッチ $128$ で学習する。

<span id="table-01"></span>

![Table 1. Comparison of various Medusa-2 models. The first section reports the details of Medusa-2, including accelerate rate, overhead, and quality that denoted the average scores on the MT-Bench compared to the original models. The second section lists the speedup ($S$) of SpecDecoding and Medusa, respectively.](../../papers/medusa/table-01.png)

**表 1.** Medusa-2 各モデルの比較。前半は高速化率、オーバーヘッド、原モデルに対する MT-Bench 平均品質、後半は SpecDecoding と Medusa の速度向上 ($S$) を示す。

<span id="figure-04"></span>

![Figure 4. Effectiveness of numbers of candidate tokens for decoding introduced by trees (default number of candidate token for decoding is 1 when using KV cache). Left: The acceleration rate for randomly sampled dense tree settings (blue dots) and optimized sparse tree settings (red stars). Right: The speed (tokens/s) for both settings. The trend lines indicate that while the acceleration rate remains relatively stable for sparse trees, there is a notable decrease in speed as the candidate tokens increases.](../../papers/medusa/figure-04.png)

**図 4.** 木が導入する候補トークン数の効果。左はランダムな密木（青点）と最適化疎木（赤星）の高速化率、右は速度。疎木の高速化率は比較的安定するが、候補増加で速度は低下する。

**結果。** [表 1](#table-01)は GPT-4 の 0〜10 点評価で各 Medusa-2 の高速化率、オーバーヘッド、品質を比較する。Vicuna-33B は高速化率が低いが品質は同等で、非公開学習データと自己蒸留データの不一致が原因と考える。品質は蒸留で揃ってもヘッド分布は学習集合からずれ得る。Vicuna 系列には開源ドラフトによる投機的デコードも適用した（[第 9 節](#section-9)）。

結果は規模拡大と自己蒸留における速度と性能の複雑な関係を示す。Medusa-2 は出力品質を保ちながら効率を高める可能性があり、LLM とヘッドの共同最適化に有望である。

<span id="section-3-3"></span>

### 3.3 アブレーション

<span id="section-3-3-1"></span>

#### 3.3.1 木構造注意の構成

Medusa-2 Vicuna-7B を使い、MT-Bench の writing と roleplay で木構造注意の動機と性能を調べる。

[図 4(a)](#figure-04)ではランダムな密木（[第 2.1.2 節](#section-2-1-2)、青点）と最適化疎木（[第 2.3.3 節](#section-2-3-3)、赤星）を比較する。64 ノード疎木は 256 ノード密木より高速化率が高い。一方、複雑な木は線形層と注意の行列積を増やし、[図 4(b)](#figure-04)の速度を下げる。高速化率は対数的に伸び、木の拡大とともに鈍る。増分がオーバーヘッド未満なら全体は遅くなる。詳細は[第 12 節](#section-12)。

<span id="figure-05"></span>

![Figure 5. Performance comparison of Medusa using proposed typical sampling. The model is fully fine-tuned from Vicuna-7B. The plot illustrates the acceleration rate and average scores on the writing and roleplay (MT-Bench) with a fixed temperature of 0.7 for 3 different settings: greedy sampling and random sampling (RS) plotted as the star and the dot, and typical sampling curves under different thresholds.](../../papers/medusa/figure-05.png)

**図 5.** 典型サンプリングを使う Medusa の比較。Vicuna-7B から全体を微調整し、温度 0.7 で writing/roleplay の高速化率と平均点を示す。星は貪欲、点はランダム (RS)、曲線はしきい値別の典型サンプリング。

<span id="section-3-3-2"></span>

#### 3.3.2 典型採択のしきい値

Medusa-2 Vicuna 7B を MT-Bench [Sto23e] の writing/roleplay で調べる。[Hew22] に従い $\alpha=\sqrt{\epsilon}$ とし、$\epsilon$ を 0.01 から 0.25 まで 0.01 刻みで変える。値を上げると品質は上がるが高速化率は下がる。創造性を要する課題ではランダムが貪欲を上回り、典型サンプリングも $\epsilon$ 増加時に同等となる。

<span id="table-02"></span>

![Table 2. Comparison of Different Settings of Vicuna-7B. Quality is obtained by evaluating models on MT-Bench using GPT-4 as the judge (higher the better).](../../papers/medusa/table-02.png)

**表 2.** Vicuna-7B の各設定。品質は GPT-4 を判定者として MT-Bench で評価し、高いほど良い。

<span id="section-3-3-3"></span>

#### 3.3.3 二段階微調整の有効性

[表 2](#table-02)では、ヘッドのみの Medusa-1 が品質を保って 2.18x、二段階学習（[第 2.2.2 節](#section-2-2-2)）の Medusa-2 が品質を保って 2.83x 高速化する。直接共同微調整は品質を落とす。したがって Medusa-2 は品質を維持し Medusa-1 より高速化できる。

<span id="table-03"></span>

![Table 3. Impact of Techniques on Speedup](../../papers/medusa/table-03.png)

**表 3.** 各技法が速度向上へ与える影響

<span id="section-4"></span>

## 4 考察

Medusa は予測ヘッドで複数トークンを同時生成し、逐次制約を回避して LLM 推論を 2.3〜2.8 倍速める。単純、パラメータ効率的、既存系へ統合しやすく、専用ドラフトを要しない。典型採択は棄却サンプリングの複雑さを除きつつ妥当な出力を保つ。二つの効率的学習法は各モデルとプロンプトで高品質を維持する。[表 3](#table-03)に技法の発展と効果をまとめる。

簡単のためバッチ 1 を扱ったが、提案は大バッチにも一般化でき、発表後に TensorRT や Huggingface TGI が対応した。

## 謝辞

本プロジェクトに大きく貢献した次の方々に深く感謝する。

- LLM 提供に貴重な洞察をくれた Zhuohan Li。彼の vLLM プロジェクトは実に見事である。
- 初期段階を形作る重要な議論に参加した Shaojie Bai。
- Tianle に切断サンプリングを紹介し LLM 提供の研究を勧めた Denny Zhou。
- LLM 提供のメモリ帯域幅制約を Tianle に指摘した Yanping Huang。
- Vicuna の規模別学習手順を明確にした Lianmin Zheng。

Jason D. Lee は NSF CCF 2002272、NSF IIS 2107304、NSF CAREER Award 2144994 の支援に謝意を表する。Deming Chen は UIUC の AMD Center of Excellence の支援に謝意を表する。

<span id="section-5"></span>

## 5 影響声明

LLM 推論を高速化する新手法 Medusa は、社会、技術、倫理に広い影響をもたらす。本節で詳しく検討する。

<span id="section-5-1"></span>

### 5.1 社会的・技術的影響

- **AI の利用可能性と民主化**：LLM の効率を高め、より多くの利用者と組織が先端 AI を使えるようにする。教育、医療、娯楽などの革新を促し得る。
- **環境への影響**：推論高速化は消費電力と炭素排出を減らし、持続可能な AI と環境保全に寄与し得る。
- **経済的影響**：先端モデルの導入費用を下げ、中小企業にも高度な AI を開放し、成長、競争、技術革新を促し得る。

<span id="section-5-2"></span>

### 5.2 倫理的考慮

- **偏りと公平性**：Medusa はバックボーンの偏りと公平性の問題を継承する。既存の偏りを維持・増幅しないか調べる必要がある。
- **透明性と説明責任**：木構造注意と複数ヘッドの複雑さは解釈性を難しくし得る。意思決定の透明性と説明責任が信頼に不可欠である。
- **安全性とプライバシー**：高速化能力は大規模な偽情報生成やサイバー攻撃の自動化に悪用され得る。倫理指針と安全策を整備・実施すべきである。

<span id="section-6"></span>

## 6 関連研究

<span id="section-6-1"></span>

### 6.1 LLM 推論高速化

LLM 推論の非効率は自己回帰デコードのメモリ帯域幅制約に由来する。バッチ推論は算術強度を高める単純な方法だが、モデルと KV キャッシュが大きなメモリを使い、大バッチを妨げる。既存法は (1) メモリ消費と転送を減らしてバッチを拡大する、(2) デコードステップを減らして直接遅延を下げる、の二つに分けられる。

**KV キャッシュ削減。** Multi-query attention [Sha19] と Grouped-query attention [Ain23] は query より少ない key/value ヘッドで KV メモリを減らし、大バッチと高利用率を可能にする [Pop22]。[Zha23g] は重要な KV トークンだけを保持し、[Kwo23] はページ式メモリ管理で断片化を減らす。

**量子化。** [Xia23] は活性とパラメータ間の再スケールで外れ値を除き、[Det22] は大半を 8 bit、少数を 16 bit の行列積にする。[Fra22] は重み列を反復的に 3/4 bit 化し、[Lin23d] は重要重みを守る活性感知量子化を提案する。[Kim23] は少数の重要重みに疎・低精度形式を用いる。

**投機的デコード。** [Lev23, Che23] は複数ステップを並列実行し、小さなドラフトが後続語を推測、LLM が一括評価して採択する。非自己回帰生成 [Xia23d] と通じるが LLM 向けである。本手法は追加ドラフトでなく原モデル自身を使い、二モデル管理を避ける。[Mia23b, Spe23] も木構造注意で複数候補を並列生成し、前者はモデル集合、後者はドラフト階層を使う。ドラフトには専用事前学習と整合が必要だが、本手法は各 Medusa ヘッドの上位予測から自己回帰なしで静的疎木を直接作り、単純かつ効率的である。木ノードが速度に与える影響も詳細に検証する。

<span id="section-6-2"></span>

### 6.2 サンプリング方式

LLM からのサンプル方法は品質を大きく左右し、直接サンプルは不整合な結果を生み得る [Pil21, Hol20]。*切断サンプリング* [Fan18, Bas21, Mei22, Hew22, Mei23] は各ステップの*許容集合*上で分布を切り、高品質で多様な標本を得る。

top-$k$ [Fan18] は上位 $k$ 語、top-$p$ [Hol20] は累積確率 $p$ を満たす最小集合を保つ。typical decoding [Mei23] は予測分布のエントロピーでしきい値を決め、[Hew22] は統一的理解を与える。

本方式も許容集合で低確率候補を除くが、出力とモデル分布の厳密な一致は要求しない。そのため品質を保ちながら多様性と効率を高められる。

<span id="section-7"></span>

## 7 実験設定

<span id="section-7-1"></span>

### 7.1 共通用語

三つの用語を明確にする。

- a) 高速化率：一デコードステップで平均何トークンをデコードするか。通常の自己回帰モデルでは 1.0。
- b) オーバーヘッド：Medusa の平均一ステップ遅延を通常モデルの値で割ったもの。
- c) 速度向上：実時間での高速化率。

したがって、速度向上 = 高速化率 / オーバーヘッドである。

<span id="section-7-2"></span>

### 7.2 共通設定

全実験で Axolotl [Axo23]、ウォームアップ付き cosine 学習率、8-bit AdamW [Det21] を使う。一層の Medusa ヘッドを $5$ 個学習し、[式 1](#equation-01)の $\lambda_k=0.8^k$ とする。Medusa-2 は LoRA [Hu21] または QLoRA [Det24] を用い、ヘッド学習率をバックボーンの $4$ 倍にする。LoRA は LM ヘッドを含む全線形層へ適用し、rank $32$、$\alpha=16$、dropout $0.05$ とする。

<span id="section-7-3"></span>

### 7.3 Vicuna 7B/13B の Medusa-1 と Medusa-2

全体バッチ $64$、ピーク学習率はバックボーン $5e^{-4}$、ヘッド $2e^{-3}$、ウォームアップ $40$ step とする。4-bit バックボーンで Medusa-1 を学習し、その結果から QLoRA の Medusa-2 を初期化する。[式 2](#equation-02)の $\lambda_0=0.2$。

<span id="section-7-4"></span>

### 7.4 Vicuna-33B と Zephyr-7B の自己蒸留

二段階でなく直接 Medusa-2 を使う。正弦スケジュールで $\theta_0$ を終盤のピークまで増やす。自己蒸留損失が小さいためバックボーン LoRA のピーク学習率は $1e^{-4}$、ウォームアップ $20$、[式 2](#equation-02)の $\lambda_0=0.01$ とする。

<span id="section-8"></span>

## 8 最適化木構造注意の可視化

[図 6](#figure-06)は Medusa-2 Vicuna-7B の疎な木である。深さ 4 は四ヘッドを表す。直積から作り、Alpaca-eval [Dub23] で測った各ヘッドの top-k 予測期待値に基づき枝刈りする。左への偏りは高確率ノードを好むことを示す。

<span id="figure-06"></span>

![Figure 6. Visualization of a sparse tree setting for Medusa-2 Vicuna-7B. The tree has 64 nodes representing candidate tokens and a depth of 4 which indicates 4 Medusa heads involved in calculation. Each node indicates a token from a top-k prediction of a Medusa head, and the edges show the connections between them. The red lines highlight the path that correctly predicts the future tokens.](../../papers/medusa/figure-06.png)

**図 6.** Medusa-2 Vicuna-7B の疎木。候補トークンを表す 64 ノード、深さ 4。各ノードはヘッドの top-k 予測、辺は接続、赤線は将来トークンを正しく予測した経路を示す。

<span id="section-9"></span>

## 9 投機的デコードの結果

Vicuna 7B/13B/33B [Chi23a] に投機的デコードを適用する。Llama-68M/160M [Mia23b]、Tiny-Llama [Zha24ab]、Vicuna 式指示調整を施した Tiny-Vicuna [Pan23a] を使う。手法 [Che23, Lev23] が非公開のためオープンソースの代替実装 [+3] で評価し、`torch.compile()` でドラフト推論を速める。

[図 7](#figure-07)では最適設定が Vicuna サイズで異なる。7B は Llama-68M の $\gamma=4$、13B は同モデルの $\gamma=3$、33B は Tiny-Vicuna (Vicuna-1B) の $\gamma=3$ が最良である。ドラフトの選択と設定は LLM サイズに合わせる必要がある。

<span id="figure-07"></span>

![Figure 7. Inference speed of various models using speculative decoding on MT-Bench. Baseline model speeds are presented by grey dotted lines for comparison. $\gamma$ denotes the draft token number.](../../papers/medusa/figure-07.png)

**図 7.** MT-Bench における各モデルの投機的デコード速度。灰点線は基準、$\gamma$ はドラフトトークン数。

<span id="section-10"></span>

## 10 全モデルの追加結果

各モデルの速度向上を[図 8](#figure-08)に示す。

<span id="figure-08"></span>

![Figure 8. Speedup of various models with Medusa-2. Medusa-2 shows significant speed improvement over all the models, while models trained with self-distillation (Zephyr-7B, Vicuna-13/33B) have weaker speedup due to the trade-off between preserving quality and boosting speed.](../../papers/medusa/figure-08.png)

**図 8.** Medusa-2 の速度向上。全モデルで大きく改善するが、自己蒸留した Zephyr-7B、Vicuna-13/33B は品質保持とのトレードオフでやや小さい。

<span id="section-11"></span>

## 11 AlpacaEval の追加結果

AlpacaEval [Li23z] でも追加実験し、MT-Bench と同様の一貫した高速化を得た。

<span id="table-04"></span>

![Table 4. Speedup results on AlpacaEval Li23z dataset.](../../papers/medusa/table-04.png)

**表 4.** AlpacaEval [Li23z] での速度向上。

<span id="section-12"></span>

## 12 ハードウェア制約と Medusa の探索・モデル化

簡略化した Llama 系列で、特にメモリ帯域幅制約が Medusa 型並列デコードへ与える影響を調べる。主要オーバーヘッドである線形層と注意行列積を特定し、A100-80GB-PCIe、A40、A6000 で FLOP/s と演算強度（FLOP/s と帯域 bytes/s の比）を測る。Medusa による変化を調べ、簡単な解析モデルとハードウェア測定からモデルサイズ、系列長、バッチサイズ別の影響を示す。

<span id="section-12-1"></span>

### 12.1 演算子のルーフラインモデル

Llama-7B/13B/33B [Tou23a] の主要な三種の行列積を各 GPU で分析する。バッチの効果を調べた [Che23g] に従うが、本稿はデコードと並列デコードに重点を置く。

[表 5](#table-05)は prefill、通常デコード、Medusa デコードの計算・空間計算量を示す。$X W_Q,X W_K,X W_V$、$Q K^\top,P V$、$X W_u,X W_g,X W_d$ を扱う。$b,s,h,i,n,d,q$ は順にバッチ、系列長、隠れ次元、中間次元、ヘッド数、ヘッド次元、候補長である。詳細は [Tou23a, Che23g]。

<span id="table-05"></span>

![Table 5. Computational and space complexity of the main operators in different phases. The table is based on the corresponding table in the report Che23g.](../../papers/medusa/table-05.png)

**表 5.** 各段階の主要演算子の計算・空間計算量。[Che23g] の対応する表に基づく。

[図 9](#figure-09)-[17](#figure-17)は三種の演算子をモデル 7/13/33B、バッチ 1〜64、系列長 128〜8192（各 2 の冪、演算子ごとに 49 設定）で測ったもの。prefill とデコードの点群は GPU やモデルサイズをまたいで近い位置に集まる。

prefill ではバッチ増加が注意行列積の FLOP/s を変えるが演算強度は変えず（[図 9](#figure-09)の縦矢印）、系列長は両方を変える（斜矢印）。デコード時はメモリ帯域幅に強く制約され、FLOP/s が増えても演算強度はほぼ不変で、自己注意の資源利用が不十分である。

prefill の線形層は概ね計算律速。デコード時の点は GPU 帯域と同傾斜で、線形層も帯域律速である。バッチ増加は並列性により FLOP/s と演算強度を上げる。線形層は新トークンのみを処理し系列長に依存しない（[表 5](#table-05)）。

<span id="figure-09"></span>

![Figure 9. The figure shows the relationship between FLOP/s and Operational Intensity for all benchmarked datapoints of Llama-7B operators on A100-80GB-PCIe. The dashed lines represent the HBM bandwidth limit (1,935GB/s) and the peak performance limit (312 TFLOP/s) Nvi20. ‘`qkv mlp`’ stands for the linear layers projecting hidden features to query/key/value features. ‘`up/gate/down`’ stands for the linear layers following the attention block. ‘`qk/pv`’ stands for the two steps of attention matrix multiplications. ‘`ar`’ stands for the decoding (autoregressive) and ‘`init`’ stands for the prefill phase.](../../papers/medusa/figure-09.png)

**図 9.** A100-80GB-PCIe 上の Llama-7B 演算子の FLOP/s と演算強度。破線は HBM 1,935GB/s とピーク 312 TFLOP/s [Nvi20]。`qkv mlp` は Q/K/V 投影、`up/gate/down` は注意後の線形層、`qk/pv` は注意行列積、`ar` は自己回帰デコード、`init` は prefill。

<span id="figure-10"></span>

![Figure 10. Llama-13B operators on A100-80GB-PCIe.](../../papers/medusa/figure-10.png)

**図 10.** A100-80GB-PCIe 上の Llama-13B。

<span id="figure-11"></span>

![Figure 11. Llama-33B operators on A100-80GB-PCIe.](../../papers/medusa/figure-11.png)

**図 11.** A100-80GB-PCIe 上の Llama-33B。

<span id="figure-12"></span>

![Figure 12. Llama-7B operators on A40.](../../papers/medusa/figure-12.png)

**図 12.** A40 上の Llama-7B。

<span id="figure-13"></span>

![Figure 13. Llama-13B operators on A40.](../../papers/medusa/figure-13.png)

**図 13.** A40 上の Llama-13B。

<span id="figure-14"></span>

![Figure 14. Llama-33B operators on A40.](../../papers/medusa/figure-14.png)

**図 14.** A40 上の Llama-33B。

<span id="figure-15"></span>

![Figure 15. Llama-7B operators on A6000.](../../papers/medusa/figure-15.png)

**図 15.** A6000 上の Llama-7B。

<span id="figure-16"></span>

![Figure 16. Llama-13B operators on A6000.](../../papers/medusa/figure-16.png)

**図 16.** A6000 上の Llama-13B。

<span id="figure-17"></span>

![Figure 17. Llama-33B operators on A6000.](../../papers/medusa/figure-17.png)

**図 17.** A6000 上の Llama-33B。

<span id="section-12-2"></span>

### 12.2 Medusa における FLOP/s と演算強度の変化

A100-80GB-PCIe 上の Llama 33B で Medusa が演算強度と FLOP/s をどう変えるか調べる。

[図 18](#figure-18)と[表 6](#table-06)はバッチ 16 の注意行列積を示す。候補増加で FLOP/s と演算強度が増す。系列長 1024、候補 64 では通常デコード比でそれぞれ $44\times$、$41\times$。[図 19](#figure-19)と[表 7](#table-07)の系列長 1024 固定では、バッチを増しても演算強度は上がらない。

[図 20](#figure-20)と[表 8](#table-08)で up/gate/down 線形層を調べる。過去トークンはキャッシュ済みなので系列長に依存しない。候補とバッチを増すと帯域律速から計算律速へ移り、Medusa が線形層の性能特性を変えることが分かる。

<span id="figure-18"></span>

![Figure 18. FLOP/s vs. Operational Intensity of attention matrix multiplication with batch size 16.](../../papers/medusa/figure-18.png)

**図 18.** バッチ 16 の注意行列積の FLOP/s と演算強度。

<span id="figure-19"></span>

![Figure 19. FLOP/s vs. Operational Intensity of attention matrix multiplication with sequence length 1024.](../../papers/medusa/figure-19.png)

**図 19.** 系列長 1024 の注意行列積の FLOP/s と演算強度。

<span id="figure-20"></span>

![Figure 20. FLOP/s vs. Operational Intensity of Linear layers.](../../papers/medusa/figure-20.png)

**図 20.** 線形層の FLOP/s と演算強度。

<span id="table-06"></span>

![Table 6. TFLOP/s & Operational Intensity of attention matrix multiplication with batch size 16 for Llama 33B on an A100 80GB PCIe.](../../papers/medusa/table-06.png)

**表 6.** A100 80GB PCIe 上の Llama 33B、バッチ 16 の注意行列積。

<span id="table-07"></span>

![Table 7. TFLOP/s & Operational Intensity of attention matrix multiplication with sequence length 1024 for Llama 33B on an A100 80GB PCIe.](../../papers/medusa/table-07.png)

**表 7.** 同モデル、系列長 1024 の注意行列積。

<span id="table-08"></span>

![Table 8. TFLOP/s & Operational Intensity of linear layers (up/gate/down) for Llama 33B on an A100 80GB PCIe.](../../papers/medusa/table-08.png)

**表 8.** 同モデルの up/gate/down 線形層。

<span id="section-12-3"></span>

### 12.3 Medusa 性能の予測

[第 3.3.1 節](#section-3-3-1)から高速化率を $\texttt{acc_rate}=0.477\log(\texttt{num_candidate})$ と近似する。バッチ 1、系列長 1024 で Llama-7B の簡略化ブロックをシミュレートし、[第 2.1.2 節](#section-2-1-2)の木構造注意で候補を並列処理する。後処理は小さいため省く。[図 21](#figure-21)では候補増加により当初は改善するが 64 超で速度向上が下がり、最適範囲がある。

[図 22](#figure-22)では系列長 1024 でバッチが 32 を超えると速度向上が低下し、負にもなり得る。線形層が帯域律速から計算律速へ移るためである。

バッチ 4 で系列長を変えた[図 23](#figure-23)では最適候補数はほぼ一定だが、長い系列ほど注意行列積の負担で性能が落ちる。線形層計算は系列長に依存しない。

候補数は一定範囲を越えると利益が減る。バッチ増加は並列性を高めるが、大きすぎれば計算律速となる。長い系列では注意を最適化する必要がある。候補数、バッチ、注意機構を均衡させることが資源利用と性能を高め、シミュレーションは設計指針として有用である。

<span id="figure-21"></span>

![Figure 21. Simulated acceleration rate, speedup, and normalized latency ablation using different numbers of candidate tokens under the setting of batch size 1 and sequence length 1024 for Llama-7B on an A100 80GB PCIe.](../../papers/medusa/figure-21.png)

**図 21.** A100 80GB PCIe 上の Llama-7B、バッチ 1、系列長 1024 で候補数を変えた高速化率、速度向上、正規化遅延のシミュレーション。

<span id="figure-22"></span>

![Figure 22. Simulated speedup with sequence length 1024 for Llama-7B.](../../papers/medusa/figure-22.png)

**図 22.** Llama-7B、系列長 1024 の速度向上シミュレーション。

<span id="figure-23"></span>

![Figure 23. Simulated speedup with batch size 4 for Llama-7B.](../../papers/medusa/figure-23.png)

**図 23.** Llama-7B、バッチ 4 の速度向上シミュレーション。

[+1]: 著者への確認によれば、この版は実験的で、Vicuna 7B/13B とは一部異なるデータを用いた。

[+2]: ここで精度は単一の第 $i$ 位トークンについて定義し、top-$i$ 精度から top-$(i-1)$ 精度を引いた値である。

[+3]: [https://github.com/feifeibear/LLMSpeculativeSampling](https://github.com/feifeibear/LLMSpeculativeSampling)

[+author-equal]: 同等の貢献。

[+author-corresponding]: 責任著者。
