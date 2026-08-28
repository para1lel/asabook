---
title: 'Conjecturing-Proving Loop'
createTime: 2026/08/28 13:11:54
permalink: /ja/papers/conjecturing-proving-loop/
pageClass: paper-reading
---

> [Kazumi Kasaura](https://www.omron.com/sinicx/en/activity/researcher/kazumikasaura/)、[Naoto Onda](https://www.ondanaoto.com/)、[Yuta Oriike](https://dblp.org/pid/410/6502)、[Masaya Taniguchi](https://tani.cc/)、[Akiyoshi Sannai](https://dblp.org/pid/220/5533)、[Sho Sonoda](https://sites.google.com/view/shosonoda/home)。2025 年 9 月 16 日に arXiv へ初投稿。現行版は 2026 年 6 月 29 日改訂の v4。[第 6 回 Natural Language Meets Logic and Machine Learning ワークショップ論文集（NALOMA）、2026 年 8 月、pp. 40-49](https://aclanthology.org/2026.naloma-1.5/)に掲載。[Discovering New Theorems via LLMs with In-Context Proof Learning in Lean](https://arxiv.org/abs/2509.14274v4)。<a href="/paper/conjecturing-proving-loop.pdf" target="_blank" rel="noopener noreferrer">原論文 PDF</a>。[DOI](https://doi.org/10.48550/arXiv.2509.14274)。[TeX ソース](https://export.arxiv.org/e-print/2509.14274v4)。正確な紙面レイアウトと参考文献については原論文 PDF を参照されたい。

## 概要

大規模言語モデル（LLM）は、形式定理証明において大きな可能性を示している。本研究では、LLM が新しい定理を発見し、検証済みの証明を生成する能力を調べる。われわれは *Conjecturing-Proving Loop*（CPL）と呼ぶパイプラインを提案する。これは数学的予想を反復的に生成し、Lean 4 での証明を試みるものである。CPL の大きな特徴は、各反復で、それまでに生成された定理と形式化された証明を LLM の条件として与える点にあり、これによってパラメータを変更せずに、文脈内学習を通じて証明戦略を改善できる。理論と実験の両面から、命題と証明を同時に生成する枠組みに比べ、CPL は証明困難な定理の発見率を高めることを示す。さらに、LLM 自身が生成して形式的に検証された出力を文脈として再利用すると、その後の証明成功率が一貫して向上することが実験で分かった。これは、自己生成した文脈による学習がニューラル定理証明に有効であることを示している。ソースコードは [https://github.com/auto-res/ConjecturingProvingLoop](https://github.com/auto-res/ConjecturingProvingLoop) で公開している。

<span id="section-1"></span>

## 1 はじめに

大規模言語モデル（LLM）は、定理証明において大きな可能性を示している。LLM は幻覚を生じることがあり、自然言語ではそれを見抜くことが難しいため、LLM で形式証明を生成し、Lean [+1] のような対話型定理証明系（ITP）で検証する研究が行われてきた。本論文では、LLM が新しい定理を発見する能力に焦点を当てる。

われわれは、数学的予想を自動生成し、Lean 4 形式で証明するパイプライン *Conjecturing-Proving Loop* を提案する。予想段階と証明段階を分離することで、同じ定理が繰り返し生成されることを避け、より難しい定理の証明を促す。言い換えると、CPL は予想／証明候補に対して*層化サンプリング*を行い、証明難度に応じて探索資源を割り当てることで、ループが簡単で短い証明ばかりに収束することを防ぐ。この層化により、命題と証明を同時にサンプリングする単純な枠組みでは発見しにくい、長く難しい証明を CPL は発見・検証できる。本論文では、この点をより詳しく理論的に議論する。

<span id="figure-01"></span>

![図 1。Conjecturing-Proving Loop の概観。ライブラリが予想器と証明器へ文脈を供給し、検証された定理がライブラリへ戻る。](../../papers/conjecturing-proving-loop/figure-01.png)

**図 1。** Conjecturing-Proving Loop の概観。予想器はライブラリを文脈として予想を生成し、証明器はその証明を試みる。証明された予想と証明は、定理としてライブラリに保存される。ライブラリは証明器にも文脈を供給する。予想器と証明器の処理はいずれも、LLM と Lean Server の対話から構成される。

本手法にはもう一つ特徴がある。証明済みの定理とその証明を含む文脈を使って、さらに定理を生成・証明することで、LLM を訓練することなく、証明戦略の文脈内学習によってより難しい証明を生成できる。GPT のようなクローズドソース LLM は、推論能力と Lean コード生成能力が近年向上しているため、われわれはそれらを予想器と証明器の両方に用いる。クローズドソース LLM にはモデルを自由に訓練できないという欠点があるが、本枠組みでは、以前に検証された証明からの文脈内学習によって LLM の証明能力を改善できる。

実験では、数学的概念を種として与えたとき、それらの重要な性質を本枠組みで再発見できるかを検証した。具体的には、Lean の数学ライブラリ Mathlib [+2] にある概念だけで定義できる一方、それ自体は Mathlib に含まれていない、いくつかの位相的概念に注目した。本枠組みを用いて、これらの概念に関する定理を生成した。その結果、数学論文で公表済みの重要な定理を本枠組みが再発見した。この定理は、予想器と証明器を分離しない単純な枠組みでは見つからなかった。さらに、証明戦略の文脈内学習が本枠組みで機能することを確認した。文脈がなければ自然言語でも LLM が証明できない重要な定理が、生成された文脈を使うことで証明された。

本論文の貢献をまとめる。第一に、数学的予想を自動生成し、Lean 4 形式で証明するパイプライン Conjecturing-Proving Loop を提案する。第二に、本枠組みが証明困難な定理の自動発見を可能にすることを、理論と実験の両面から示した。第三に、目標定理の命題が与えられる前に LLM 自身が生成した検証済みの証明を文脈として与えることで、文脈内学習によって LLM の証明能力を改善できることを確認した。

本研究は、AI によって形式数学ライブラリを自動的に拡張できる可能性も示している。形式化された数学は、自然言語で表現された数学の一部にすぎず、Mathlib のような形式ライブラリを拡張することは、数学の検証と自動化に不可欠である。一方、ライブラリに含めるべき命題の集合を、常に自然言語から得られるとは限らない。本枠組みは、与えられた概念を学習しながら、それに関する命題を生成できる。

<span id="section-2"></span>

## 2 関連研究

LLM を数学的推論に用いる研究は複数あり、自然言語 [Dee24a] と ITP 向け形式言語 [Ren25b, Lin25i] の双方が扱われている。これらは主に既存問題の解決に焦点を当て、教師ありファインチューニング（SFT）および／または検証可能報酬による強化学習（RLVR）を用いて LLM の問題解決能力を改善している。訓練データの不足を補うため、AI が解くべき問題を生成する方法も先行研究で提案されている [Ma25, Hua25a, Zha25ac]。これらの研究と本手法には二つの違いがある。第一に、本手法は意味のある定理の生成と証明に焦点を当てるのに対し、これらの研究は LLM 証明器の訓練に焦点を当てる。第二に、これらは強化学習に基づくが、本手法は文脈内学習に基づくため、クローズドソース LLM にも適用できる。

プロンプトに適切な文脈を含めると LLM の数学的推論能力が向上することは、複数の研究で報告されている [Wei22a, Zho22c, Dro22, Hu24c, Poi25]。これらの研究が手作業の例やデータベースから抽出したデータを文脈内学習の材料にするのに対し、本枠組みは文脈内強化学習 [Moe25b] と同様に、LLM 自身の出力を用いる。さらに、難しい定理の証明に使う補題を予想・証明する手法も提案されている [Tha23, Wan23k, Che25h, Bab25]。これらの研究とは異なり、われわれは証明対象の定理を LLM に与えない。LLM がその定理を発見する能力に焦点を当て、証明生成時に文脈として使うライブラリも、目標定理の命題が与えられる前に作られる。

ITP からのフィードバックを形式証明生成に利用する手法は既に提案されており [Fir23, Tha23, Lin25j]、われわれも採用している（[第 3.3 節](#section-3-3)を参照）。ただし、ここで強調するのは、別の命題の検証済み証明から戦略を学習することである。

Minimo [Poe24] は本枠組みと似ており、予想器と証明器のエージェントを同時に訓練して定理を見つける。ただし、Minimo は既存知識を使わずに数学を再発見することを目指すのに対し、本研究の目的はより実用的であり、既存の大規模言語モデルを使って定理の発見を試みる。

あるサーベイ [Zha26a] は、LLM を用いる手法も含め、定理生成について包括的かつ最新のまとめを提供している。

<span id="section-3"></span>

## 3 手法

本節では、まず枠組みの概要を示し、次に予想器と証明器のアーキテクチャを説明する。

<span id="section-3-1"></span>

### 3.1 パイプラインの概要

[図 1](#figure-01)に本枠組みを示す。Conjecturing-Proving Loop（CPL）は、予想器（LLM エージェント）、証明器（LLM エージェント）、Lean サーバ、ライブラリ（Lean コードデータ）の四つから構成される。最初に、ユーザがライブラリを初期化する。

1. 予想器は、Lean サーバにアクセスしながら、ライブラリに基づいて有効な Lean 4 形式の新しい数学的予想を生成する。
2. 証明器は、生成された予想ごとに、Lean サーバにアクセスしながら有効な証明の生成を試みる。この段階でもライブラリを文脈として使う。
3. 検証された予想と証明の組をライブラリに追加する。その後、最初の段階へ戻る。

予想段階と証明段階の詳細は、以下の各小節を参照されたい。

予想段階と証明段階を分離することで、同じ定理が繰り返し生成されることを避け、より難しい定理の証明を促す。詳しい議論は[第 4 節](#section-4)を参照されたい。

ライブラリを予想器の文脈として与える目的は、予想の重複生成を防ぎ、既に証明された定理から類推して予想を生成することである。

ライブラリを証明器の文脈として与える目的は、証明時に既に証明された定理を利用可能にし、文脈内学習によって証明戦略を学ぶことである。

<span id="section-3-2"></span>

### 3.2 予想ループ

多様な予想を生成するため、各予想段階で次の処理を行う。

1. 予想器 LLM は、現在のライブラリに続けて予想を生成する。
2. Lean サーバは、生成された各予想が構文的に有効で新しいかを確認する。検証済みの予想は証明器へ送られる。

予想の新規性は、文脈内の既存定理でその予想を証明できるかを調べる Lean の `exact?` コマンドで確認する。このコマンドは、Mathlib4 全体（Lean4 標準ライブラリ）を import し、ライブラリと検証済み予想を含む文脈で実行される。したがって、ここで確認される新規性とは、その予想が Mathlib4、生成済みライブラリ、検証済み予想のいずれにも既に存在しないことを意味する。

予想器 LLM に与えるシステムプロンプトは次のとおりである。

> あなたは mathlib4 ライブラリのコントリビュータです。与えられたライブラリに基づいて、Lean 4 形式の新しい定理の予想を生成してください。真である必要はありません。リストに既にある命題は生成しないでください。証明、アノテーション、import は含めないでください。新しい各命題は 'theorem' で始め（アノテーションなし）、':= sorry' で終えてください。また、Unicode エスケープシーケンス（例：\u2200）ではなく、標準的な数学記号（例：$\forall$、$\exists$、$\sqrt{}$）を使ってください。

<span id="section-3-3"></span>

### 3.3 証明器ループ

証明器は、生成された予想ごとに次の処理で証明を試みる。

1. 証明器 LLM が予想の証明コードを生成する。LLM が予想を証明不能と判断した場合、証明器は失敗としてループを終了する。
2. Lean サーバが生成された証明を検証する。証明が検証されれば、証明器は成功としてループを終了する。
3. 最大試行回数に達していれば、証明器は失敗としてループを終了する。そうでなければ、Lean サーバのエラーメッセージを LLM に返し、手順 1 へ戻る。

文脈は証明器と Lean サーバの双方に与えられる。したがって、証明器は文脈から証明戦略を学ぶだけでなく、文脈内の定理を補題として利用できる。

証明器 LLM に与えるシステムプロンプトは次のとおりである。

> あなたは mathlib4 ライブラリのコントリビュータです。与えられた内容の最後の定理を Lean 4 で証明してください。最後の定理の ':=' の直後に置く Lean 4 コードを書いてください。コードは 'by' で始めるか、項の式にしてください。与えられた内容の定理を補題として使ってもかまいません。証明に 'sorry' を使わないでください。定理が証明不能だと判断した場合は、証明ではなく空文字列を返してください。ほかのテキストは含めないでください。

実験では最大試行回数を $16$ とした。

<span id="section-3-4"></span>

### 3.4 ベースライン

比較のため、LLM が定理と証明を一度に生成する単純ループ（SL）でも、これらの概念に関する定理を生成した。最初にユーザがライブラリを初期化する。CPL と異なり、この単純ループのベースラインでは予想器と証明器を分離せず、単一のループを次のように実行する。

1. LLM は Lean サーバにアクセスして検証しながら、ライブラリに基づいて Lean 4 形式の命題と証明を生成する。
2. 前の手順が成功した場合、生成された命題と証明の組をライブラリに保存する。その後、手順 1 へ戻る。

手順 1 は証明器ループに似ている。詳細は次のとおりである。

1. LLM が Lean で命題とその証明を生成する。
2. Lean サーバが生成内容を確認する。検証されれば、成功としてループを終了する。
3. 最大試行回数に達していれば、失敗としてループを終了する。そうでなければ、Lean サーバのエラーメッセージを LLM に返し、手順 1 へ戻る。

LLM に与えるシステムプロンプトは次のとおりである。

> あなたは mathlib4 ライブラリのコントリビュータです。与えられたライブラリに基づき、Lean 4 形式の新しい定理と証明を生成してください。Lean 4 コード以外は出力しないでください。生成コードは与えられたライブラリの後に続くもので、定理の命題と証明だけを含めてください。variable、section、namespace など、theorem 以外の宣言は出力しないでください。ライブラリに既に存在する定理は生成しないでください。新しい定理は 'theorem' で始めてください（アノテーションなし）。与えられたライブラリ内の定理を証明の補題として使ってもかまいません。証明に 'sorry' を使わないでください。また、Unicode エスケープシーケンス（例：\u2200）ではなく、標準的な数学記号（例：$\forall$、$\exists$、$\sqrt{}$）を使ってください。

証明器ループと同様に、最大試行回数は $16$ とした。

<span id="section-4"></span>

## 4 理論

以下の理由により、SL と CPL では生成定理の分布が異なると予想される。命題と証明を一度に生成すると、生成定理の分布は命題の分布と証明成功率の両方に依存する。一方、命題を生成してから複数回証明を試みると、定理の分布は証明可能な命題の分布に近づき、証明成功率の影響は小さくなる。

より形式的には、LLM が命題 $T$ を生成する確率分布を $s(T)$、LLM が $T$ の証明生成に成功する確率を $r(T)$ とする。SL は、命題と証明を順に生成し、証明が正しければ両者を出力する単純化された過程としてモデル化する。CPL も単純化し、命題を生成した後、有効な証明が得られるか、試行回数が $N$ に達するまで証明生成を試みる過程としてモデル化する。簡単のため、文脈の影響は無視する。

定理 $T$ の生成確率は、SL では $s(T)r(T)$ に比例し、CPL では $s(T)\left(1-(1-r(T))^N\right)$ に比例する。したがって、$N$ が増えると CPL の定理分布は、証明可能な命題（$r(T)>0$ となる $T$）の分布へ近づき、証明が難しい定理も生成されやすくなる。

一方、SL で一つの定理を発見するために必要な証明試行回数の期待値は $E_\mathrm{SL}:=\left(\mathbb{E}_{T\sim s}[r(T)]\right)^{-1}$ であるが、CPL では

$$
E_\mathrm{CPL}:=\frac{\mathbb{E}_{T\sim s}\left[(1-(1-r(T))^N)r(T)^{-1}\right]}{\mathbb{E}_{T\sim s}\left[1-(1-r(T))^N\right]}
$$

となる。命題 $T$ が生成されたとき、$T$ の証明に成功する確率は $1-(1-r(T))^N$、試行回数の期待値は $(1-(1-r(T))^N)r(T)^{-1}$ だからである。 [+3]

$\left(1-(1-r)^N\right)r^{-1}$ は $r$ について減少するので、チェビシェフの和不等式から

$$
\mathbb{E}_{T\sim s}\left[(1-(1-r(T))^N)\right]
\leq \mathbb{E}_{T\sim s}\left[(1-(1-r(T))^N)r(T)^{-1}\right]\mathbb{E}_{T\sim s}[r(T)].
$$

よって $E_{\mathrm{SL}}\leq E_{\mathrm{CPL}}$ であり、CPL が SL より少ない定理しか生成しない理由を説明できる。

証明試行回数を固定したとき、CPL で定理 $T_0$ が SL より生成されやすくなる条件は次のとおりである。SL では、命題を一度生成したとき $T_0$ を発見する確率は $s(T_0)r(T_0)$ であり、証明試行回数は常に $1$ である。CPL では、命題を一度生成したとき $T_0$ を発見する確率は $s(T_0)(1-(1-r(T_0))^N)$、証明試行回数の期待値は $\mathbb{E}_{T\sim s}\left[(1-(1-r(T))^N)r(T)^{-1}\right]$ である。$s(T_0)\ll 1$ なので、求める条件は

$$
\frac{1-(1-r(T_0))^N}{\mathbb{E}_{T\sim s}\left[(1-(1-r(T))^N)r(T)^{-1}\right]} > r(T_0),
$$

と近似でき、$s(T_0)$ に依存しない。$r(T_0)>0$ ならば、これは

$$
(1-(1-r(T_0))^N)r(T_0)^{-1}
> \mathbb{E}_{T\sim s}\left[(1-(1-r(T))^N)r(T)^{-1}\right]
$$

とも書ける。$\left(1-(1-r)^N\right)r^{-1}$ は $r$ について減少するため、証明成功率が十分に低い証明可能な定理は、CPL でより生成されやすい。

<span id="section-5"></span>

## 5 実験

本枠組みが研究水準の定理を再発見できることを示し、本枠組みで文脈内学習が有効に働くことを確認した。

実験用スクリプトと生成ライブラリは [https://github.com/auto-res/ConjecturingProvingLoop](https://github.com/auto-res/ConjecturingProvingLoop) に保存している。

<span id="section-5-1"></span>

### 5.1 設定

実験では、一般位相空間論における比較的周辺的な概念である半開性、$\alpha$-開性、preopenness に注目する。これらを Lean 4 形式で定義した次のファイルを初期ライブラリに用いた。

```lean
import Mathlib
import Aesop

namespace Topology

variable {X : Type*} [TopologicalSpace X]

def P1 (A : Set X) : Prop :=
  A ⊆ closure (interior A)

def P2 (A : Set X) : Prop :=
  A ⊆ interior (closure (interior A))

def P3 (A : Set X) : Prop :=
  A ⊆ interior (closure A)
```

P1、P2、P3 はそれぞれ「半開」「$\alpha$-開」「preopen」を表し、LLM が既存知識を使うことを防ぐため匿名化している。これらを選んだ理由は、Mathlib に既にある概念だけで定義できる一方、それ自体はまだ Mathlib に含まれておらず、数学的な重要性が既に認められて研究されているものの、その性質を LLM が知識として持つほど有名ではないためである。

次の定理を目標に設定し、生成できるかどうかに注目した。

> *二つの P2（$\alpha$-開）集合の共通部分は P2（$\alpha$-開）である*

この定理は、$\alpha$-開集合が別の位相をなすことを証明するうえで最も難しい部分であるため重要である（[Nja65] の命題 2）。少なくとも、実験で使った LLM の知識からこの定理を素朴に導けないことは確認済みである。[第 5.3.3 節](#section-5-3-3)を参照されたい。生成ライブラリが目標定理を含むかどうかは、生成ライブラリの後に定理の命題を置き、`exact?` コマンドで証明が補完されるかを調べて確認する。ライブラリに、その定理と自明に同値か、より強い命題が含まれていれば補完は成功する。この方法により、Lean サーバが認識できる範囲で、定理の定式化の違いを許容できる。

CPL と SL の両方で GPT-o3 [+4] を用いた。CPL と SL のそれぞれについて、API 使用量が $14000000$ トークンに達するまで定理を生成する操作を $20$ 回行い、ライブラリを生成した。

<span id="section-5-2"></span>

### 5.2 結果

CPL では平均 106 個の定理が生成され、**20 回中 5 回、目標定理が発見された**。SL では平均 328 個の定理が生成されたが、**20 回のいずれでも目標定理は生成されなかった**。Fisher の正確確率検定（$p=0.024$）によると、CPL は目標定理を生成しやすい。

この定理の生成証明例を[第 7 節](#section-7)に示す。この証明は Njåstad の原証明と異なっており、LLM が独自にこの証明を発見したことを示唆する。

SL のほうが多くの定理を生成する一方、CPL は証明困難な定理を生成しやすいという結果は、[第 4 節](#section-4)の議論と整合する。これをさらに検証するため、生成定理の証明長を測定した。[図 2](#figure-02)は、CPL と SL が生成した定理について、証明長（文字数）の分布を示す。CPL が SL より長い証明を生成できることが分かる。証明の長さと難しさには正の関係があることが知られている [+5] [Wu25p, Son26b]。したがって、この結果は理論解析と整合する。

<span id="figure-02"></span>

![図 2。CPL と SL が生成した定理の証明長ヒストグラム。](../../papers/conjecturing-proving-loop/figure-02.png)

**図 2。** 本枠組みと単純ループの枠組みが生成した定理について、証明長（文字数）の分布。

<span id="section-5-3"></span>

### 5.3 文脈を与える効果

上述した CPL の効果を独立に確認するため、追加実験を行った（[第 5.3.1 節](#section-5-3-1)）。

また、生成ライブラリを文脈として証明器へ与えることで、証明能力が向上することも確認した（[第 5.3.2 節](#section-5-3-2)および[第 5.3.3 節](#section-5-3-3)）。

<span id="section-5-3-1"></span>

#### 5.3.1 文脈内学習を用いない生成

文脈内学習の影響を除いた CPL と SL の違いを見るため、種ファイルだけを文脈として定理を生成した。すなわち、CPL と SL の両方について、API 使用量が $3000000$ トークンに達するまで、最初の単一ループを独立に数回実行した。

<span id="figure-03"></span>

![図 3。文脈を使わずに CPL と SL が生成した定理の証明長ヒストグラム。](../../papers/conjecturing-proving-loop/figure-03.png)

**図 3。** 文脈を使わない場合に、本枠組みと単純ループの枠組みが生成した定理について、証明長（文字数）の分布。

重複を含め、CPL では $309$ 個、SL では $941$ 個の定理が生成された。生成証明の分布を[図 3](#figure-03)に示す。分布のずれが観察される。Kolmogorov-Smirnov 検定によると、CPL は SL より長い証明を生成する傾向があり、その $p$ 値は $1\times 10^{-13}$ である。

目標定理は CPL と SL のどちらでも生成されなかった。以下の実験結果も参照されたい。

<span id="section-5-3-2"></span>

#### 5.3.2 生成定理の再証明

まず、CPL が生成した全定理を、二つの設定で再証明しようとした。一つは、証明対象の定理が生成される前に作られたライブラリを文脈に含む設定、もう一つは概念の定義だけを文脈に含む設定である。その結果、文脈がある設定では**定理の 99%（2106/2123 個）**が証明されたのに対し、文脈がない設定では**定理の 91%（1935/2123 個）**しか証明されなかった。McNemar 検定によると、この差は $4\times 10^{-35}$ の p 値で統計的に有意である。したがって、文脈は LLM の証明能力を改善する。

<span id="section-5-3-3"></span>

#### 5.3.3 Alpha-開集合の共通部分に対する証明能力

さらに、目標定理が生成された $5$ 個の文脈それぞれについて、目標定理の再証明を $16$ 回試みた。（この定理が生成されるまでに生成された定理数の平均は $49$ である。）比較のため、生成ライブラリなしでも $80$ 回再証明を試みた。手順は証明器ループと同じだが、システムプロンプトを次のように変更した。

> あなたは mathlib4 ライブラリのコントリビュータです。与えられた内容の最後の定理を Lean 4 で証明してください。最後の定理の ':=' の直後に置く Lean 4 コードを書いてください。コードは 'by' で始めるか、項の式にしてください。与えられた内容の定理を補題として使ってもかまいません。証明に 'sorry' を使わないでください。定理が偽だと判断した場合は、証明ではなく空文字列を返してください。ほかのテキストは含めないでください。

証明を返さない条件が「証明不能」から「偽」へ変更されている点に注意されたい。

その結果、**生成ライブラリを文脈に含む設定では再証明に $7$ 回成功したが、ライブラリなしの設定では $80$ 回すべて失敗した**。これは、証明器が文脈内学習を通じて、文脈なしでは証明できなかった定理を証明する能力を獲得することを示唆する。

[第 7 節](#section-7)に示すこの定理の生成証明は、ほかの生成定理を補題として使っていない。したがって、生成ライブラリは証明用の補題集としてではなく、証明戦略の文脈内学習に使われた。

加えて、LLM（GPT-4o [+6] と GPT-o3）に対し、概念の定義を含む文脈を与え、自然言語（英語）でこの定理を $16$ 回証明するよう求め、応答を手作業で確認した。自然言語の実験では、次のシステムプロンプトを使った。

> 次の定理を証明してください。定理が偽だと判断した場合は、証明の代わりに "False" を返してください。

LLM に与えた証明対象の命題は次のとおりである。

> 位相空間において、集合がその内部の閉包の内部の部分集合であるとき、その集合を alpha-開と呼ぶ。任意の二つの alpha-開集合の共通部分は alpha-開である。

その結果、GPT-4o は命題が偽だと $10$ 回誤って述べ、誤った証明を $6$ 回生成した。GPT-o3 は誤った証明を一度も生成しなかったが、定理が偽であると常に誤って判断した。GPT-4o の判断の大半がこの定理を偽としていることは、GPT の知識にこの定理が含まれていなかったことを示唆する。GPT-4o が生成した、欠落のある証明例を[第 8 節](#section-8)に示す。

<span id="section-6"></span>

## 6 結論と今後の課題

数学的予想を自動生成し、Lean 4 形式で証明するパイプライン Conjecturing-Proving Loop を提示した。本枠組みが研究水準の定理を再発見できることを示した。また、証明戦略の文脈内学習が本枠組みで有効に働くことも確認した。

本研究で取り上げた命題は、予想すること自体は比較的自然であった。今後は予想生成過程を改良し、より深く洞察に富む数学的命題を生成する必要がある。たとえば、LLM を数学理論の未探索領域へ導く手法を組み込むことが考えられる。

## 謝辞

本研究は、JST Moonshot R&D Program JPMJMS2236、JST BOOST JPMJBY24E2、JST CREST JPMJCR2015、JSPS KAKENHI 24K21316、24K16077、Advanced General Intelligence for Science Program（AGIS）、RIKEN TRIP initiative の支援を受けた。

<span id="section-7"></span>

## 7 対象定理の生成証明

::: details 証明
```lean
theorem P2_inter {X : Type*} [TopologicalSpace X] {A B : Set X} : P2 A → P2 B → P2 (A ∩ B) := by
  intro hP2A hP2B x hx
  rcases hx with ⟨hxA, hxB⟩
  -- Auxiliary open neighbourhoods furnished by `P2 A` and `P2 B`.
  set UA : Set X := interior (closure (interior A)) with hUA
  set UB : Set X := interior (closure (interior B)) with hUB
  have hUA_open : IsOpen UA := by
    simpa [hUA] using (isOpen_interior :
      IsOpen (interior (closure (interior A))))
  have hUB_open : IsOpen UB := by
    simpa [hUB] using (isOpen_interior :
      IsOpen (interior (closure (interior B))))
  have hxUA : x ∈ UA := by
    have : x ∈ interior (closure (interior A)) := hP2A hxA
    simpa [hUA] using this
  have hxUB : x ∈ UB := by
    have : x ∈ interior (closure (interior B)) := hP2B hxB
    simpa [hUB] using this
  -- Combine the two neighbourhoods.
  have hO_open : IsOpen (UA ∩ UB : Set X) := hUA_open.inter hUB_open
  have hxO : x ∈ UA ∩ UB := ⟨hxUA, hxUB⟩
  -- Main claim: the intersection lies in the relevant closure.
  have hO_sub : (UA ∩ UB : Set X) ⊆ closure (interior (A ∩ B)) := by
    intro y hy
    rcases hy with ⟨hyUA, hyUB⟩
    -- `y` is simultaneously in the closures of `interior A` and `interior B`.
    have hy_clA : y ∈ closure (interior A) := by
      -- `UA ⊆ closure (interior A)`
      have hsub : (UA : Set X) ⊆ closure (interior A) := by
        intro z hz
        have hz' : z ∈ interior (closure (interior A)) := by
          simpa [hUA] using hz
        exact interior_subset hz'
      exact hsub hyUA
    have hy_clB : y ∈ closure (interior B) := by
      have hsub : (UB : Set X) ⊆ closure (interior B) := by
        intro z hz
        have hz' : z ∈ interior (closure (interior B)) := by
          simpa [hUB] using hz
        exact interior_subset hz'
      exact hsub hyUB
    -- Show that every open neighbourhood of `y` meets `interior (A ∩ B)`.
    have : y ∈ closure (interior (A ∩ B)) := by
      apply (mem_closure_iff).2
      intro V hVopen hyV
      -- First refinement, intersect with `UB`.
      have hV1_open : IsOpen (V ∩ UB) := hVopen.inter hUB_open
      have hyV1 : y ∈ V ∩ UB := ⟨hyV, hyUB⟩
      -- Obtain a point in `interior A`.
      rcases (mem_closure_iff).1 hy_clA (V ∩ UB) hV1_open hyV1 with
        ⟨a, ⟨haV, haUB⟩, haIntA⟩
      -- `a` is now in `V`, `UB`, and `interior A`.
      have ha_clB : a ∈ closure (interior B) := by
        have hsub : (UB : Set X) ⊆ closure (interior B) := by
          intro z hz
          have hz' : z ∈ interior (closure (interior B)) := by
            simpa [hUB] using hz
          exact interior_subset hz'
        exact hsub haUB
      -- Second refinement, intersect with `interior A`.
      have hW_open : IsOpen (V ∩ interior A) := hVopen.inter isOpen_interior
      have haW : a ∈ V ∩ interior A := ⟨haV, haIntA⟩
      -- Obtain a point in `interior B`.
      rcases (mem_closure_iff).1 ha_clB (V ∩ interior A) hW_open haW with
        ⟨z, ⟨hzV, hzIntA⟩, hzIntB⟩
      -- `z` lies in `V`, `interior A`, and `interior B`.
      have hzIntAB : z ∈ interior (A ∩ B) := by
        -- The set `interior A ∩ interior B` is open and contained in `A ∩ B`.
        have hS_open : IsOpen (interior A ∩ interior B) :=
          isOpen_interior.inter isOpen_interior
        have hS_sub : (interior A ∩ interior B : Set X) ⊆ A ∩ B := by
          intro w hw
          rcases hw with ⟨h1, h2⟩
          exact ⟨interior_subset h1, interior_subset h2⟩
        have hS_to : (interior A ∩ interior B : Set X) ⊆
            interior (A ∩ B) :=
          interior_maximal hS_sub hS_open
        have hzS : z ∈ interior A ∩ interior B := ⟨hzIntA, hzIntB⟩
        exact hS_to hzS
      exact ⟨z, hzV, hzIntAB⟩
    exact this
  -- Use the neighbourhood just constructed.
  have hNhd : (UA ∩ UB : Set X) ∈ 𝓝 x :=
    hO_open.mem_nhds hxO
  have h_mem :
      x ∈ interior (closure (interior (A ∩ B))) :=
    (mem_interior_iff_mem_nhds).2
      (Filter.mem_of_superset hNhd hO_sub)
  simpa using h_mem
```
:::

<span id="section-8"></span>

## 8 生成された誤った証明の例

以下は、二つの alpha-開集合の共通部分が alpha-開であるという定理について、GPT-4o が生成した「証明」の一例である。

::: details 証明
与えられた定理を証明するため、まず位相空間で集合が alpha-開であることの意味を定義する。

$(X, \tau)$ を位相空間、$A \subseteq X$ とする。集合 $A$ が alpha-開であるとは、

$$
A \subseteq \mathrm{int}(\overline{\mathrm{int}(A)}),
$$

が成り立つことをいう。ここで $\mathrm{int}(A)$ は $A$ の内部、$\overline{\mathrm{int}(A)}$ は $\mathrm{int}(A)$ の閉包を表す。

任意の二つの alpha-開集合の共通部分が alpha-開であることを示す。$A, B \subseteq X$ を二つの alpha-開集合とする。このとき定義より、次が成り立つ。

1. $A \subseteq \mathrm{int}(\overline{\mathrm{int}(A)})$
2. $B \subseteq \mathrm{int}(\overline{\mathrm{int}(B)})$

$A \cap B$ も alpha-開、すなわち

$$
A \cap B \subseteq \mathrm{int}(\overline{\mathrm{int}(A \cap B)})
$$

であることを示す必要がある。

まず、$\mathrm{int}(A \cap B) \subseteq \mathrm{int}(A) \cap \mathrm{int}(B)$ に注意する。

閉包と内部の演算はいずれも単調なので、

$$
\overline{\mathrm{int}(A \cap B)} \subseteq \overline{\mathrm{int}(A) \cap \mathrm{int}(B)} \subseteq \overline{\mathrm{int}(A)} \cap \overline{\mathrm{int}(B)}.
$$

さらに内部を取ると、

$$
\mathrm{int}(\overline{\mathrm{int}(A \cap B)}) \subseteq \mathrm{int}(\overline{\mathrm{int}(A)} \cap \overline{\mathrm{int}(B)}) \subseteq \mathrm{int}(\overline{\mathrm{int}(A)}) \cap \mathrm{int}(\overline{\mathrm{int}(B)}).
$$

$A \subseteq \mathrm{int}(\overline{\mathrm{int}(A)})$ および $B \subseteq \mathrm{int}(\overline{\mathrm{int}(B)})$ という仮定から、

$$
A \cap B \subseteq \mathrm{int}(\overline{\mathrm{int}(A)}) \cap \mathrm{int}(\overline{\mathrm{int}(B)}) \supset \mathrm{int}(\overline{\mathrm{int}(A \cap B)}).
$$

したがって、

$$
A \cap B \subseteq \mathrm{int}(\overline{\mathrm{int}(A \cap B)}).
$$

これは $A \cap B$ が実際に alpha-開であることを示す。よって、任意の二つの alpha-開集合の共通部分は alpha-開である。

したがって、定理は真である。
:::

この証明には明らかに欠落がある。最後から二番目の関係から最後の関係を導く際に、部分集合関係の向きが逆転しているためである。

[+1]: [https://lean-lang.org/](https://lean-lang.org/)

[+2]: [https://github.com/leanprover-community/mathlib4](https://github.com/leanprover-community/mathlib4)

[+3]: $(1-(1-r)^N)r^{-1}$ は実際には多項式なので、$r=0$ での値を $N$ とみなす。

[+4]: [https://platform.openai.com/docs/models/o3](https://platform.openai.com/docs/models/o3)。GPT は現在 5.2 まで公開されているが、o3 は本研究の開始時点で最新の版だった。実験で使った概念と定理は o3 を前提に設計されており、GPT-5 は当初から性能が高く実験に適さなかったため、採用しなかった。

[+5]: ここで「難度」とは Lean で有効な証明を生成する難しさを指し、「証明長」とは Lean コードの長さを指す。これらは自然言語における難しさや長さと必ずしも一致しない。

[+6]: [https://platform.openai.com/docs/models/gpt-4o](https://platform.openai.com/docs/models/gpt-4o)
