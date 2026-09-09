---
title: 'AutoGraph: Imperative Code, Graph Performance'
createTime: 2026/09/09 10:55:00
permalink: /ja/papers/autograph/
---

> [Dan Moldovan](https://dblp.org/pid/209/9551.html)、[James M. Decker](https://dblp.org/pid/198/0881.html)、[Fei Wang](https://dblp.org/pid/52/3194-46.html)、[Andrew A. Johnson](https://dblp.org/pid/186/0067-2.html)、[Brian K. Lee](https://dblp.org/pid/228/7805.html)、[Zachary Nado](https://dblp.org/pid/228/7785.html)、[D. Sculley](https://dblp.org/pid/s/DSculley.html)、[Tiark Rompf](https://dblp.org/pid/19/7373.html)、[Alexander B. Wiltschko](https://dblp.org/pid/209/9851.html)。Google Brain（米国マサチューセッツ州ケンブリッジ）および Purdue University。連絡先は Tiark Rompf と Alexander B. Wiltschko。キーワード：機械学習、SysML。2018 年 10 月 16 日に arXiv へ初投稿であり、現行版は 2019 年 3 月 26 日投稿の v2。[Proceedings of Machine Learning and Systems 1（MLSys 2019）](https://proceedings.mlsys.org/paper_files/paper/2019/hash/e31cef6b735cf838db79202dbee7b093-Abstract.html)に掲載。[AutoGraph: Imperative-style Coding with Graph-based Performance](https://arxiv.org/abs/1810.08061)。<a href="/paper/autograph.pdf" target="_blank" rel="noopener noreferrer">原文 PDF</a>。[arXiv DOI](https://doi.org/10.48550/arXiv.1810.08061)。[TeX ソース](https://export.arxiv.org/e-print/1810.08061)。正確な印刷レイアウトと参考文献については、原文 PDF を正本とする。

## 概要

書きやすい機械学習コードと、スケーラブルまたは高速に実行できる機械学習コードとの間には、トレードオフがあると考えられている。機械学習では、Autograd や PyTorch のような*命令型*ライブラリは書きやすい一方、インタプリタ処理のオーバーヘッドが大きく、本番環境やモバイル環境へのデプロイも容易ではない。TensorFlow や Theano のような*グラフベース*のライブラリは、プログラム全体の最適化の恩恵を受け、幅広くデプロイできるものの、複雑なモデルの表現が煩雑になる。本稿では、ソースコード変換を介した Python のステージドプログラミングが、両方の利点を取り込んで、これら二つのライブラリ設計パターンの中間点を提供する仕組みを述べる。重要な着想は、動的ディスパッチと同様に、型に依存するすべての判断を実行時まで遅延させることである。これらの原則を、TensorFlow ライブラリでのプログラミング体験を改善するソフトウェアシステム AutoGraph として具現化し、ネイティブな TensorFlow グラフと比較して性能を損なうことなく使いやすさが向上することを示す。また、本システムがバックエンド非依存であり、TensorFlow グラフにはない特性を備えた別の IR をターゲットにできることも示す。

<span id="section-1"></span>

## 1 機械学習のプログラミングパラダイム

ニューラルネットワークなどの機械学習（ML）モデルが、翻訳や画像認識といった産業上重要な多くの問題で最先端の性能を示すにつれ、機械学習に特化したプログラミングプラットフォームが広く採用されるようになった。この利用拡大を支えるため、新しい ML モデルを構築するプラットフォームの開発が急速に進んでいる。これらのプラットフォームは、*グラフベース*プログラミングと*命令型*プログラミングという二つの主要なパラダイムに従う。両者はそれぞれ *Define-and-run* と *Define-by-run* とも呼ばれる [Tok15]。

TensorFlow や Theano のような*グラフベース*のシステムは、高水準言語（通常は Python）を使って、計算を表す低水準の中間表現（IR）をメタプログラミングする [Aba16b, The16]。TensorFlow の場合、この IR は、データセンター内へ自動的に分散したり、GPU や TPU などのアクセラレータハードウェアで実行したり、モバイル端末やウェブサーバーへデプロイしたりできる表現を提供し、プログラム全体の最適化も利用できる。計算上の利得は大きいが、その代償として開発者の認知的負荷が増える。

PyTorch や Autograd [Pas17, Mac15] のような*命令型*プログラミングシステムは、ユーザーコードを直接実行しながら、自動微分またはコンパイルに用いるユーザープログラムの表現を逐次構築する。TensorFlow も「eager execution」による命令型スタイルのコーディングをサポートしており、そこではユーザーが書いた Python コードが、グラフを構築せずに TensorFlow カーネルを即座に実行する。この種のシステムでは従来の命令型コーディングの利点を享受できるが、プログラム最適化、スケーラブルな計算、可搬性の機会は減少する。

これらの手法の違いは、条件分岐やループなど、データ依存の制御フローを必要とするモデルで特に明瞭になり、こうした制御フローは、強化学習、系列モデル、その他多くの新興研究分野における最先端手法で重要である。*命令型*プラットフォームでは、条件分岐やループのようなデータ依存の制御フロー演算に従来の構文を用い、慣用的でネイティブな Python の制御フローを書ける。しかし、この手法はプログラム全体を最適化する機会を減らし、自動微分のために実行のたびにトレースし直す必要がある。*グラフベース*のプラットフォームはこの問題を回避する一方、データ依存の制御フローに従来の Python 構文を使うことを認めず、代わりにあらゆるデータ依存制御フローを関数形式で表現するよう求める。これは、Python が制御フローの実行遅延をネイティブにサポートしていないために必要となる。

*グラフベース*と*命令型*のプログラミングは、直交し独立したパラダイムとして語られることが多いが、本稿では、*命令型*の使いやすさを保ちながら*グラフベース*の性能と可搬性も得られる、両者の長所を兼ね備えた手法を提示する。この手法は、コードを特化した IR へ変換できること、またその IR が、速度、メモリ、数値安定性の最適化や多様なプラットフォームへのデプロイ可能性など、プログラマーに実質的な利点をもたらすことを前提とする。ただし、多くの IR と同様に、直接プログラミングするのは煩雑であるとも仮定する。TensorFlow は広く使われ、堅牢な IR を備えるため、本稿の議論の大半は TensorFlow グラフに焦点を当てる一方、評価（[第 9.1 節](#section-9-1)）では、この手法がいかなるバックエンドからも完全に独立しており、実際、コード生成エンジンのターゲットに別のバックエンドを選ぶことで、TensorFlow の IR では容易に表現できないプログラムも表現できることを示す。

本稿の貢献は次のとおりである：

- *グラフベース*システムの性能と可搬性を保ちながら、*命令型* ML システムの表現力をユーザーに提供する新しい方法論を提案する。
- 静的解析と*ソースコード変換*（SCT）を用い、この方法論を Python で実証する。
- これらの解析とコード変換により、実行時の型情報に基づいてディスパッチするステージドプログラミングを Python で可能にする。ほとんどの場合、追加の注釈は必要ない。
- AutoGraph と呼ぶ本システムを使い、慣用的な Python を TensorFlow Graph IR へ変換する。AutoGraph が他のバックエンドをターゲットにできるよう一般化されており、再入可能な関数呼び出しなど TensorFlow Graph IR にない機能をサポートする Lantern IR へ Python コードを変換できることを示す。
- 最適化済み IR へ低水準化され、手書きの代替実装と同等の速さで動く複雑な ML プログラムを、本システムによってユーザーが容易に表現できることを示す。

<span id="section-2"></span>

## 2 関連研究

既存の多くのシステムや手法も、性能を低下させずに ML モデルを定義できる、使いやすいプログラミングインターフェースの提供を目指している。その一例が Open Neural-Network eXchange（ONNX）形式 [Onn18] であり、最適化と高性能計算に重点を置く一般的な多数のバックエンドをターゲットにできる、多数の高水準フロントエンド向け API を備えた IR を提供する。この IR は、多くの*命令型*システムと同じく、トレースを通じて生成される計算グラフとして提示される。ONNX は、*命令型*システムと*グラフベース*システムの仲介役に IR を使える可能性を示しているが、トレースによるグラフ抽出ではデータ依存の制御フローを捕捉できないため、制御フロー情報が失われるおそれがある。

近年の別の手法として、PyTorch の Torch Script フレームワーク [Pyt18] がある。AutoGraph と同様に Python AST の変換に基づくが、重要な相違点がいくつもあり、なかでも動的形状グラフ上の形状伝播を超えるステージングがない点が際立つ。Torch Script と AutoGraph のより詳細な比較は[第 10 節](#section-10)に示す。Myia システム [Mer18] は Torch Script と同様の機能を提供し、ユーザーが Python で数値コードを記述すると、それが Python AST とは異なるグラフベース IR へ解析される。JANUS [Jeo19] は、Python インタプリタを変更し、Python バイトコードから TensorFlow グラフコードへの JIT コンパイラのように動作する。これに対し、AutoGraph はソースからソースへの変換を行う独立したライブラリとして動作する。

ステージドプログラミングや多重ディスパッチを用いて遅延実行を容易にする試みには、長い歴史がある。代表例には、Lightweight Modular Staging の型ベース遅延実行モデル [Rom10]、Lua と Terra を組み合わせて高性能な数値コードをステージングする手法 [Dev13]、Julia の多重ディスパッチシステム [Bez12] がある。Python でコード書き換えを実装または利用するライブラリは限定的に使われており、プライバシーと機密性を考慮した Jeeves システム [Yan16a]（MacroPy [Hao13] に依存）や、Python に埋め込まれた Lisp 方言である Hy システム [Hy18] などがある。しかし、これらの手法はいずれも、大幅に変更しなければ単独では Python 言語に適さない。

このほかにも、異なる機能を備えた多様な ML フレームワークが提案されている。Lantern [Wan18e, Wan18f] は、プログラミング言語研究の概念（限定継続とマルチステージプログラミング）を応用し、表現力の高い*グラフベース* ML フレームワークを実装した。Tangent [Mer17a] は SCT を用いて自動微分を行う。Dynet [Neu17] は、計算を自動的にバッチ化する動的バッチングランタイムを備えた define-by-run システムである。MXNet [Che15b] は、異なる構文を用いることで *define-by-run* と*グラフベース*の両方を提供する。chainer [Tok15] と、Autograd ライブラリを Lua に移植した torch-autograd [Tor18] は、どちらも純粋な define-by-run システムである。Numba [Lam15] は、注釈付き Python 関数を実行時に機械語へ変換する。

<span id="section-3"></span>

## 3 TensorFlow のプログラミング

TensorFlow ソフトウェアプログラミングシステムは、ML 実務者、とりわけ大規模な学習とデプロイに注力する人々の間で広く使われるようになった [Hal18]。モデルアーキテクチャとデータ例がプログラムのライフサイクルの異なる時点で利用可能になるため、ML プログラムは自然に別々のステージで実行され、TensorFlow はこれらのステージを明示する。TensorFlow のユーザーは、まず実行する計算の表現を構築し、その後プログラム内で、その計算を実行するよう指定しなければならない。この表現には、容易に最適化、分散、デプロイできることから、データフローグラフが使われる。このプログラミングモデルは直感的でない場合があり、使い勝手の難しさやバグにつながり、とりわけ制御フローを指定する場合に問題が顕著である。たとえば、低水準化された IR に含めるべき制御フロー構文がある一方、計算を IR にステージングするかどうかを指定するための制御フロー構文もある。一般的なコーディングパターンの一つは、モデルのハイパーパラメータを使って計算を条件付きでステージングすることである：

```python
# Conditional on bool not added to graph
if HParams.nonlin == 'relu':
  x = tf.nn.relu(x)
else:
  x = tf.nn.tanh(x)
```

しかし、制御フローの別の用途では、データに依存して実行することが意図されている：

```python
# Conditional on Tensor added to graph
x = tf.cond(tf.reduce_sum(x) > 0,
  lambda: x * x, lambda: x)
```

上のコードでは、条件文を関数形式で表現することで、グラフ内でデータに依存して実行できるようにしている。しかし、これは美的にも実用上も Python の命令型スタイルと衝突する。制御フローを入れ子にしたり、`continue` や `break` など、ほかの Python の慣用表現を使ったりする必要があると、困難はさらに増す。代わりに、次のように書きたい：

```python
# Conditional on Tensor - staged
if tf.reduce_sum(x) > 0:
  x = x * x
```

そして、これが自動的に関数形式へ変換されるようにしたい。この変換は、数値型を使う式に対してだけ行われることが望ましい。通常の Python 真偽値で分岐する条件文（たとえば、上のハイパーパラメータの例）は、ステージングせず命令的に実行されるべきである。

<span id="section-4"></span>

## 4 演算子オーバーロードの拡張

TensorFlow では、複雑なプログラムのデータフローグラフをメタプログラミングするのは難しいが、*演算子オーバーロード*によって容易になる。たとえば、ユーザーは `tf.add(a, b)` と明記せず、単に `a + b` を使える。これは、Python が言語の一部をプログラマーによってオーバーロードできるようにしているためである。Python の演算子オーバーロードでは、TensorFlow の `Tensor` 型のようなカスタムクラスが、二項演算子（例：`+,*,-,%,/,^,~`）で使われたときの振る舞いや要素アクセスなど、一部の既定機能を上書きできる。[+1]

```python
# Because Python lets us write this ...
class Tensor(_TensorLike):
  def __add__(self, right):
    return tf.add(self, right)

# ... we can write this
import tensorflow as tf
a = tf.constant(3)
b = tf.constant(4)
c = a + b
```

これは Python 言語の強力な機能だが、残念ながらオブジェクトやクラスのメソッドにしか及ばず、現代的な ML モデルの構築に必要なプログラミング構文は含まれない。たとえば、Python では条件分岐の振る舞いをオーバーロードできない。

```python
# We can write if statements...
if cond:
  ans = true_fn()
else:
  ans = false_fn()

# ... but we cannot overload them
def __if__(self, cond, true_fn, false_fn):
  if cond:
    return true_fn()
  else:
    return false_fn()
```

制御フロー構文をオーバーロードできれば、*命令型*プログラムは、従来は不可視だったループや条件文も含め、ユーザーコードの完全な表現を生成できる。*グラフベース*プログラムでは、`__if__`、`__for__`、`__while__` や Python 言語のその他の有用な部分に非標準のオーバーライドを提供できるため、煩雑な関数形式でプログラムの制御フローを書くようユーザーに求める必要がなくなる。

この制約を回避するため、関数全体に SCT を適用し、Python 言語の非局所的な部分をオーバーロードできるようにする。本稿ではこのシステムの具体的な実装として、SCT を使い、ユーザーが慣用的な Python を書きながら低水準 IR をターゲットにできるようにする *AutoGraph* を説明する。

<span id="section-5"></span>

## 5 実用 ML システムのためのステージドプログラミング

任意の Python 構文をオーバーロードする機能を用い、*命令型*スタイルの ML プログラムの性能を高め、反対に*グラフベース* ML プログラムを簡潔にするため、AutoGraph と呼ぶステージドプログラミングシステムを構築した。

AutoGraph では、ユーザーは慣用的な命令型スタイルの Python でプログラムしながら TensorFlow グラフの利点も享受でき、[リスト 1](#listing-01)に示す Python 関数デコレーターとして、単一関数の API をユーザーに公開する。

<span id="listing-01"></span>

```python
import autograph as ag

# AutoGraph converts whole
# functions via a decorator...
@ag.convert()
def f(x):
  if x > 0:
    x = x * x
  return x

# ... into a form where control flow
# and other idioms are overloadable
def new_f(x):
  def if_true():
    x_1 = x
    x_1 = x_1 * x_1
    return x_1
  def if_false():
    return x
  x = ag.if_stmt(
    ag.gt_(x, 0), if_true, if_false)
  return x
```

**リスト 1.** AutoGraph は上側のコードを下側のコードへ自動変換する（簡略化した例）。

AutoGraph は、`if`、`for`、`while` 文などの制御フローが任意に入れ子になっていたり、`break` 文や `continue` 文を含んでいたりしても処理できる。

AutoGraph システムは SCT を介して条件分岐とループをオーバーロードできるため、Python の既定動作から外れることができる。同じ方式の SCT を使えば、一部の文をオーバーロードしながら、他の文では Python の意味論を保つという選択も可能である。このため、本機能は Python 開発者一般に有用なツール、または新しい言語実装が搭載を検討しうる機能になると考えている。[第 3 節](#section-3)の条件分岐の例のように、TensorFlow でステージングする制御フローとしない制御フローの両方を透過的にサポートするには、真偽述語の型に応じて `if` 文の振る舞いを変えなければならない。

<span id="section-6"></span>

## 6 「動的ディスパッチ」による Python のステージドプログラミング

Python でオーバーロード可能な制御フローを実現できるため、`ag.if_stmt` の非標準実装を書くことで、その既定動作を再定義できる。Python の真偽値が条件分岐の述語として使われる場合は、通常の意味論で条件分岐を実行したい。一方、TensorFlow の Tensor や、その他の特化した数値型が与えられた場合は、より特化したコードをステージングしたい。`ag.if_stmt` の簡略版を[リスト 2](#listing-02)に示す。

<span id="listing-02"></span>

```python
def if_stmt(cond, body, orelse):
  if is_tensor(cond):
    return tf.cond(cond, body, orelse)
  elif cond:
    return body()
  else:
    return orelse()
```

**リスト 2.** AutoGraph による条件文オーバーライドの簡略版。

この実行時の判断は、オブジェクト指向プログラミングで一般的な動的*メソッド*ディスパッチに似ているため、*動的ディスパッチ*と呼ぶ。動的ディスパッチによって重要なのは、ML コードにおける制御フローの二つの一般的な用途、すなわちハイパーパラメータの値に応じて分岐またはループする「マクロプログラミング」モードと、制御フローをターゲット IR へ低水準化するデータ依存モードとをシームレスに切り替えられることである。

同じ論理を、`ag.for_stmt` 関数と `ag.while_stmt` 関数に相当する処理で、`for` ループと `while` ループにも適用する。また、通常は TensorFlow グラフと互換性のない `print` 文をオーバーライドする機能も提供するが、これは `print` が情報を即座に記録するのに対し、ここではグラフの実行時に値を記録したいためである。

`break` 文や `continue` 文など、Python のネイティブ構文の一部には TensorFlow での直接的な表現がないことに注意されたい。このため、プログラムの意味論に影響を与えず、これらの文を完全に除去するコード変換が必要になる。それぞれの文を同等の TensorFlow 構文へ低水準化することで実現する。たとえば `continue` は、追加の変数と条件分岐を用いて低水準化される。

動的ディスパッチ手法には、実行時の追加オーバーヘッドが伴う。実際、通常のステージングされない Python 計算に AutoGraph を使えば、処理は遅くなる。しかし、Python ランタイムとは別に実行できる低水準 IR をターゲットにするため、このオーバーヘッドは償却される。

**一般的な手順。** 関数の変換は次のように進む：

1.  関数のソースコードを読み、利用可能ならクロージャ変数を取得する。
2.  Python のバージョン間にある小さな差異を抽象化し、ソースコードを Python AST に解析する。
3.  ソースコードを複数のパスで変換する。各パスは、次の二つの主要な段階からなる。
    1.  後述する静的解析。実際の変換で利用できる付加情報を AST に注釈する。
    2.  AST 変換。各変換が特定の Python 慣用表現を処理する。個々の変換は後述する。
4.  最終的な AST を出力コードへシリアライズする。
5.  新しい出力コードを Python 関数として読み込み、元の関数のクロージャ変数に対応するシンボルを動的に結び付ける。

**静的手法との比較。** Python コードから計算グラフを静的に抽出することは可能だが、そのためには厳格な制約が必要になる。Torch Script [Pyt18] のようなシステムは、Python の限定的なサブセットである DSL として、これらの制約を課すことを選んでいる。一方、AutoGraph の大きな設計判断は、元の Python インターフェースを可能な限りユーザーが利用できるようにすることである（この制約については[第 10 節](#section-10)で論じる）。さらに、束縛時解析のため、何らかの代用的な静的型システム（例：静的型注釈）を要求しないかぎり、静的手法だけに依存すると Python でのステージドプログラミングができなくなる。任意の型を扱う動的環境でステージドプログラミングを可能にするには慎重な検討が必要だが [Dec19]、配列ベースの型（テンソル）を中心に据える TensorFlow を主なバックエンドとして選んだことで、実装上の難しさの一部は大幅に軽減される。これについては[第 7 節](#section-7)で詳しく論じる。

<span id="section-7"></span>

## 7 コード解析と変換

自明に変換できるのは Python の一部だけであり、ステージドプログラミングに必要なオーバーロードを可能にするには、ユーザー提供コードを大幅に書き換えなければならない。たとえば、ループと条件分岐は関数形式に書き換え、非局所制御フロー文は低水準化する必要がある。これらの書き換えは、データフロー解析やその他のプログラム構造解析を利用して行う。また、書き換えを複数の特化したパスに分割する。

<span id="section-7-1"></span>

### 7.1 データフロー解析

各特化パスの前には、複数のデータフロー解析パスを実行する。以下、実行順に説明する。

**制御フローグラフの構築。** 標準的な手続き内制御フローグラフ（CFG）が、複数の静的解析を支える。

**修飾名の解決。** シンボルの概念を `a.b` のような複合名まで拡張するため、修飾名の抽象化を作成する。たとえば、修飾名 `a.b` は、おおよそ AST の `Attribute(name=Name('a'), attr='b')` に対応する。

**活動性解析。** 各文が読み取り、変更するシンボルの一覧を AST ノードに注釈する。書き込みと見なすのは直接的な変更だけである。たとえば、文 `a.b = c` では `a.b` は変更されたと見なすが、`a` は変更されたとは見なさない。活動性解析は、字句スコープ、その入れ子関係（例：親スコープ）、そこに含まれるシンボルも追跡する。

**到達定義解析。** この標準的なデータフロー解析は、各名前に到達する定義の特定に役立つ注釈を付ける。さらに、特定の文への入口で定義済みのシンボル一覧にも注釈を付ける。

**生存性解析。** この標準的なデータフロー解析は、条件分岐などの複合文を含む特定の文について、その入口または出口で生存しているシンボルを特定する。

<span id="section-7-2"></span>

### 7.2 コード変換パス

AutoGraph は、通常は独立している複数の AST 変換パスからなる拡張可能なシステムを使ってコード変換を行う。たとえば、ある変換パスは `if` 文をオーバーロード可能な関数形式へ書き換える。別のパスは `break` 文を、新しいループ述語と追加の条件分岐へ低水準化する。この仕組みにより、今後さらに多くの Python 慣用表現への対応を追加しやすくなる。

現在の変換には、適用順に次のものが含まれる：

**ディレクティブ。** AutoGraph のコンパイルディレクティブとして機能する特定の関数呼び出しを特定し、該当する AST ノードに注釈する。そのようなディレクティブの一例が `ag.set_loop_options` である。

**Break、Continue、Return 文。** 実際には三つの別々のパスだが、性質は非常によく似ている。いずれの場合も、対応する文を条件分岐または拡張したループ条件へ低水準化する。

```python
# Before conversion
if cond:
  return f(x)
return g(x)

# After conversion
if cond:
  return_value = f(x)
else:
  return_value = g(x)
return return_value
```

**Assert 文。** オーバーロード可能な関数形式へその場で変換する。

**リスト。** リストリテラルや `append`、`pop` 関数呼び出しを含むリストの慣用表現を、それぞれの演算をステージングできるカスタム関数（例：`ag.list_append` と `ag.list_pop`）でオーバーロードする。

配列計算には、標準 Python ライブラリに存在しない追加の慣用表現、すなわち stack 演算が必要になる。AutoGraph は、他のオーバーロードと整合する方法でオーバーロード可能な `ag.stack` 関数を提供する。リストアクセス（例：`l[i]`）と変更は、スライス演算子を扱う別の変換パスへ先送りする点に注意されたい。

**スライス。** Python では、ユーザークラスでスライス演算子（`__setitem__`、`__getitem__`）をオーバーロードできる。しかし、スライスへの書き込み演算には対象を変更する意味論がある。そこで、現在 TensorFlow が要求する値意味論を使うよう、スライスへの書き込みを書き換える。たとえば、`x[i] = y` はその場で `x = ag.setitem(x, i, y)` へ変換される。スライスの読み取り演算は機械的に変換する。

**関数呼び出し。** すべての関数呼び出しをオーバーロードする。呼び出される関数の特性と変換設定に応じて、オーバーロードは対象関数を動的に変換するか、そのまま呼び出すか、新しい関数に置き換える。たとえば、組み込み関数 `print` は `tf.print` へ変換されることがある（詳細は[第 16 節](#section-16)を参照）。

```python
# Before conversion
def f(a, x):
  return a(x)

# After conversion (simplified)
def f(a, x):
  return ag.converted_call(a, x)
```

**制御フロー。** この変換パスは、すべての局所制御フローを、オーバーロード可能で等価な関数形式に置き換える。

`if` 文は状態を持たないため、その関数形式は、文内で変更されるすべての変数を返す零項関数を使って表現できる。

```python
# Before conversion
if x > 0:
  x = x * x

# After conversion (simplified)
def true_fn():
  return x * x
def false_fn():
  return x
x = ag.if_stmt(x > 0, true_fn, false_fn)
```

Python では、制御フロー文の本体内でシンボルを定義（すなわち初回代入）し、後で利用できる点に注意されたい。条件分岐の枝が実行されたかどうかによって、シンボルが未定義になりうるコードを書くことができる。しかし、関数版の条件演算子は、条件分岐がいずれかの枝で変更しうるシンボルを常に設定する。未定義の意味論を模擬するため、特殊な値を使って変数の「未定義」状態を具象化する。これは現在 Python の意味論から外れているが、「未定義」のシンボルが使われる前に検証し、明示的に削除することで解決する予定である。

`while` ループと `for` ループは状態を持つため、その関数形式では、引数と戻り値がループ内で変更される変数（ループの状態）を表す関数が必要になる。

```python
# Before conversion
while x > eps:
  x = f(x)

# After conversion (simplified)
def loop_test(x):
  return x > eps
def loop_body(x):
  return f(x)
x = ag.while_stmt(
  loop_test, loop_body, (x,))
```

`for` 文も同様に処理する。`if` 文と同じく、`while` ループと `for` ループは本体内でシンボルを定義することがある。ループ本体が一度も実行されなければ、それらのシンボルは未定義のままになる。これについても、ループへの入口で未定義（生存性解析で特定）となるシンボルに、特殊な「未定義」値を使って対処する。

オーバーロードされた制御フローは動的ディスパッチを使う（[第 16 節](#section-16)を参照）。

**三項条件式。** 三項演算子 `x if cond else y` は、その場で関数形式 `ag.if_stmt(cond, x, y)` へ変換される。

**論理式。** 二項および単項論理式は、従来の演算子オーバーロード（例：`<` 演算子に対する `__lt__`）で処理できる。しかし、互換性上の理由から `Tensor` はすべての演算子をサポートしていない（たとえば `__eq__` はサポートされない）。そのため、特定の二項演算子と単項演算子を、その場でオーバーロード可能な関数形式に置き換える。たとえば、`a and b` は `ag.and_(a, b)` へ置き換えられる。

**関数ラッパー。** この変換パスは、関数ブロック全体を追加の定型コードで包む。たとえば、描画されたグラフの可読性を高める TensorFlow の*名前スコープ*を作成するために必要な呼び出しを収容する。さらに、関数ラッパーには、使いやすさを改善するため特定のエラーを捕捉する専用エラーハンドラーが含まれる。

<span id="section-8"></span>

## 8 TensorFlow を越えて：別のバックエンド

このコード変換のバックエンドが TensorFlow だけなら、TensorFlow の制約は AutoGraph にもそのまま及ぶ。しかし、メタプログラミングの性質上、AutoGraph の SCT は、多様なバックエンドをターゲットにするため容易に利用できる。すでに述べたとおり、TensorFlow の欠点の一つは、グラフ内の再入可能関数、ひいては再帰モデルを扱えないことである。AutoGraph が実装する汎用 SCT 方法論の有用性を示すため、再帰モデルを記述するグラフを生成できる Lantern [Wan18e, Wan18f] という新しい ML フレームワークのプロトタイプをターゲットに選ぶ。

**Lantern IR。** Lantern バックエンドは、数値演算を記述する Lisp 風の S 式を効率的な C++ コードへ変換する。重要なのは、Lantern が関数再帰やインライン関数定義など、TensorFlow のグラフ仕様にはなく、一部の最先端 ML 言語モデルには不可欠なプログラミング機能をサポートすることである。追加のコード変換パスでサポートされる Lantern の S 式 IR をターゲットにして、AutoGraph の汎用性を実証する。

**関数と再帰のステージング。** モデル内の関数を扱うため、`__def_staging(function, *args)` と `__call_staging(function, *args)` という二つの新しい関数を導入する。これらは、生成される S 式に、それぞれ関数定義または関数呼び出しを出力する。AutoGraph が提供する遅延 API により、既知のパラメータに関して S 式 IR 内の生成関数を特殊化できる。関数呼び出しと定義の特殊化は、AutoGraph に既存のディスパッチおよびオーバーロード機構で処理されるため、追加の変更を必要としない点に注意されたい。生成した計算グラフで関数を定義し呼び出せることで、再帰モデルの定義と実行に必要なインターフェースが得られる。

これを実証するため、Python $\to$ S-Expr $\to$ C++ のエンドツーエンド例を示す。まず、次の Python の再帰関数を取り上げる：

```python
@ag.convert()
def tree_prod(base, tree):
  if not tree.is_empty:
    l = tree_prod(base, tree.left)
    r = tree_prod(base, tree.right)
    return l * r * tree.value
  else:
    return base
```

Lantern をターゲットにできる変更を加えると、これから次の Python コードが生成される（提示用に簡略化）：

```python
def run(base, tree):
  def tree_prod(base, tree):
    def true_fn():
      return base

    def false_fn():
      l = __call_staged(tree_prod,
        base, tree.left)
      r = __call_staged(tree_prod,
        base, tree.right)
      return l * r * tree.value
    ag.if_stmt(tree.is_empty,
      true_fn, false_fn)
  __def_staged(tree_prod, base, tree)
  return __call_staged(tree_prod, base,
    tree)
```

ステージングされた関数を正しく生成するには、定義中の関数へ最終的に渡される引数を `__def_staged` に渡さなければならない点に注意されたい。これを実行すると S 式コードが生成され、そのコードが Lantern への入力として渡され、Lantern は内部計算を行い、最終的に次の C++ コードを生成して実行する：

```cpp
double Snippet(double base, Tree tree) {
  auto rec = [&](Tree tree,
  function<double(double)> cont,
  double base) {
    double grad = 0.0;
    if (!tree.is_empty) {
      auto cont_l = [&](double x1) {
        double sub_grad = 0.0;
        auto cont_r = [&](double x2) {
          double x3 = tree.value;
          double x4 = cont(x1 * x2 * x3);
          double x5 = x3 * x4;
          sub_grad += x2 * x5;
          return x1 * x5;
        };
        grad += rec(tree.R, cont_r, base);
        return sub_grad;
      };
      grad += rec(tree.L, cont_l, base);
    } else
      grad += cont(base);
    return grad;
  };
  return rec(tree,
    [&](auto x){return 1.0;}, base);
}
```

図示したように、再帰関数をステージングするには、生成される C++ コードも再帰的でなければならない（`rec` 関数がそれを示す）。誤差逆伝播を処理するため、生成される C++ コードはかなり複雑に見える。誤差逆伝播はコールバックを介して実装され、その詳細は [Wan18e, Wan18f] を参照できる（コード内では `cont`、`cont_l`、`cont_r` と記された継続として現れる）。

<span id="section-9"></span>

## 9 評価

複数の観点から AutoGraph の有用性を検証した。第一に、性能上の不利益を生じさせずに、データ依存の制御フローに頼る ML コードの可読性を AutoGraph が改善できるかを問うた。第二に、確率的勾配降下法（SGD）の学習プロセス全体など、通常は TensorFlow グラフの外に置かれる計算を、AutoGraph でグラフ IR 内へ移せるかを検証した。第三に、別の IR をターゲットにすることで、TensorFlow グラフがサポートしない機能を使った高性能コードを AutoGraph で生成できるかを検証した。さらに、Attention を用いたニューラル機械翻訳、Sequence-to-sequence、MAML メタ学習、L-BFGS 最適化など、より複雑なアルゴリズムの追加例も用意した。これらは[第 15 節](#section-15)に示す。

**RNN セル。** 次のコード断片は、単純な入力に対して TensorFlow 組み込みの `tf.dynamic_rnn` 関数と同一の結果を生成し、同程度の速度で動作する RNN モデルの実装である。

```python
def dynamic_rnn(rnn_cell, input_data,
  initial_state, sequence_len=None):
  input_data = tf.transpose(input_data,
    (1, 0, 2))
  outputs = []
  ag.set_element_type(outputs, tf.float32)
  state = initial_state
  if sequence_length is None:
    max_len = tf.shape(input_data)[0]
  else:
    max_len = tf.reduce_max(sequence_len)
  for i in tf.range(max_len):
    prev_state = state
    output, state = rnn_cell(input_data[i],
      state)
    state = tf.where(
      i < sequence_len,
      state,
      prev_state)
    outputs.append(output)
  outputs = ag.stack(outputs)
  outputs = tf.transpose(outputs,
    (1, 0, 2))
  return outputs, state
```

この簡潔で読みやすい実装を、[第 12 節](#section-12)の等価なグラフ版と比較されたい。

<span id="table-01"></span>

![系列長とバッチサイズごとの RNN セルのスループット](../../papers/autograph/table-01.png)

**表 1.** RNN セルの性能（千例/秒）

TensorFlow 公式の `tf.dynamic_rnn` 実装を、手書きのグラフベース実装、および上のコード断片を AutoGraph でグラフへ変換した実装の双方と比較した。各試行では、バッチサイズと系列長を変えながら、隠れ状態のサイズが 256 の RNN を実行した。5 回のウォームアップを行い、その後 100 回の試行の平均と標準偏差を報告する。すべての例で、各試行は 1 回の `tf.Session.run()` 呼び出しとして実行した。すべてのベンチマークは、デュアルスレッドの 6 コア Intel Xeon E5-1650 CPU で実行した。AutoGraph の利用はコードの可読性を高め、性能への影響はごくわずかである。

**グラフ内学習。** 通常、単一の学習ステップを表す TensorFlow グラフは、TensorFlow の外にある Python の学習ループで繰り返し実行される。この方法が使われるのは、TensorFlow グラフ内で制御フロー演算子を使うのが難しいためだが、追加の計算オーバーヘッドが生じる。ここでは AutoGraph を使い、計算グラフとして完全に実装された学習ループを示す。確率的勾配降下法（SGD）を用いて MNIST 上で単一の線形層を学習し、その性能をほかの複数の実装と比較した。第一の手法は、NumPy や PyTorch に似た TensorFlow の命令型実行モードである TensorFlow Eager である。第二の手法は、従来型の TensorFlow 学習プロセスである。第三の手法は、TensorFlow の `while_loop` API を用いて実装したグラフ内学習ループである。

<span id="table-02"></span>

![四つの実行戦略における学習スループット](../../papers/autograph/table-02.png)

**表 2.** モデルと学習ループ

各試行は、バッチサイズ 200 で 1000 学習ステップからなる。1 回のウォームアップを行い、その後 10 回の試行の平均と標準偏差を報告する。グラフ内学習ループの例では、1000 学習ステップ全体を 1 回の `tf.Session.run()` 呼び出しで実行する。ほかの例では、各学習ステップを個別の `tf.Session.run()` 呼び出しとして実行する。Python ループ内で単一学習ステップのグラフを繰り返し実行する方法（従来手法）は、eager スタイルのコードより 75% 高速である。学習プロセス全体を TensorFlow グラフへ移すと、さらに約 30% の高速化が得られた。

<span id="section-9-1"></span>

### 9.1 AutoGraph + Lantern：TreeLSTM

Stanford Sentiment データセット [Soc13] 上で動作する感情分類用 TreeLSTM モデルを、[Tai15] に従って評価した。このモデルは、左右の部分木を再帰的に埋め込み、BiLSTM コアを介して埋め込みベクトルを結合することで、文の構文木を埋め込む。次に文全体の埋め込みを MLP へ渡し、感情を予測する。このモデルは、再帰関数を使って PyTorch で容易に表現でき、Python の再帰関数をターゲットとする AutoGraph でも容易に表現できる。最終的に生成された C++ コードを、学習効率の観点で PyTorch 実装と比較した。「実環境」に近い実行時間を得るため、この実験は Ubuntu 16.04 上で、デュアルコア AMD A9-9410 Radeon CPU @ 1.70 GHz と 8 GB の SODIMM Synchronous 2400 MHz RAM を備えたノート PC を用い、単一スレッドで実行した。

Lantern をターゲットとする TreeLSTM の AutoGraph 実装は、PyTorch 実装の約 2.38 倍の性能を示した。本システムは毎秒約 36.75 SGD ステップを達成したのに対し、PyTorch 実装は毎秒 15.41 ステップだった。再帰モデルのバッチ化が難しいため、両システムともバッチサイズ 1 を用いた。

<span id="table-03"></span>

![PyTorch と Lantern を用いた AutoGraph における TreeLSTM の学習スループット](../../papers/autograph/table-03.png)

**表 3.** Lantern をターゲットとする TreeLSTM

<span id="section-10"></span>

## 10 考察

ソースコード変換の方法論を開発する作業は、決して機械的ではない。設計上の判断がいくつも存在し、最終的には表現力、性能、可搬性の面で異なる結果をもたらしうる。本節では、それらの判断の一部を論じ、現在の制約も含め、AutoGraph の現状がどのように形作られたかを明らかにする。AutoGraph のエラー処理については、[第 13 節](#section-13)で詳しく論じる。

**機能としてのエンジニアリング実践。** AutoGraph に実装したコード変換パスは非局所的であり、互いに複雑な形で相互作用しうる。たとえば、深く入れ子になった `for` ループと `if` 文を変換すると、入れ子の各階層間にあるデータフローの相互作用が表面化する。信頼できるシステムを構築するため、エンジニアリングのベストプラクティスを幅広く活用した。たとえば、すべての静的解析、コード変換、ユーティリティ関数を徹底的にユニットテストしている（AutoGraph の約 2.2 万行のコードのうち、テストが >50% を占める）。さらに、機能間の相互作用をエンドツーエンドの参照テストで検証する。AutoGraph システムへの変更には、すべてのユニットテストと参照テストの通過を求め、すべてのコードを少なくとも 1 人のエンジニアが、正しさ、可読性、スタイルガイドラインへの準拠について手作業でレビューする。経験的には、このテストとレビューを重視した開発実践によって、意外で微妙なバグが数多く見つかり、AutoGraph のように複雑なライブラリでも、比較的容易に保守、拡張できる状態を保てた。さらに、Python ソースコードを操作する有用なユーティリティを多数構築し、開発を簡素化した（[第 14 節](#section-14)で説明する）。

**ステージドプログラミングを実装する別の手法。** SCT に代わる手法として、TensorFlow グラフへ対応付けられる、Python プログラム用の非標準実行意味論を備えた新しい Python インタプリタを構築することも考えられ、実際、AutoGraph の初期提案はまさにそれを目指していた。しかし、非標準の Python インタプリタでは、機械学習コードで変更を必要としない部分まで含め、Python 言語のあらゆる側面を再実装しなければならない。

近年の Myia システム [Mer18] が採った戦略のように、Python を独自の中間表現へ解析することもできる。この中間表現は、その後 Python へ逆変換するか、専用 VM で実行できる。実際、この戦略は Lantern と連携する本手法に似ており、AutoGraph は、IR として S 式を生成するよう元の Python ソースコードを変更し、その S 式を Lantern が受け取る。

変換後に Python コードを出力するという選択には、いくつかの利点がある。未サポートのコード慣用表現でも、プログラムの意味論に影響しないなら変換をそのまま通過できる。これにより、従来の TensorFlow コードへの対応が簡単になる。さらに、生成されたコードをユーザーが検査し、変更することさえできる。

**Torch Script と AutoGraph の比較。** ONNX と同様、PyTorch の Torch Script フレームワーク [Pyt18] では、ユーザーが後の評価に備えてモデルを保存できると同時に、二つの新しいデコレーターを備えた、ほぼネイティブな Python という、さらに高水準のプログラミングインターフェースを提供する。`torch.jit.trace` と `torch.jit.script` というデコレーターは、方法こそ異なるが、慣用的な Python から Torch Script コード（最終的な計算グラフの IR として使われる Python のサブセット）を生成する。

`torch.jit.trace` デコレーターは、その名のとおり、トレースによって計算グラフを抽出する。これにより、形状に完全に特殊化された Torch Script コードが生成され、高度に最適化されたモデルが可能になる（また、将来のコンパイラにとって扱いやすいターゲットにもなる）。しかし、Torch Script のトレースには ONNX と同じ欠点があり、Torch Script の開発者が明言するように、「トレースが関数とモジュールを正しく記録できるのは、それらがデータ依存でない場合（たとえば、テンソル内のデータに対する条件分岐を持たない場合）に限られる……」。

一方、Torch Script の `torch.jit.script` デコレーターは、装飾された Python 関数を Torch Script コードへ直接変換するため、データ依存の制御フローを利用できる。これは AutoGraph のソースコード変換モデル（[第 6 節](#section-6)で詳述）に似て見えるが、二つの方法論には重要な違いがいくつもある。Torch Script は本質的に PyTorch ランタイムへ結び付いているため、ほかの特化型または高速化された ML バックエンドとは併用できない。さらに、`torch.jit.script` はすべての処理をコンパイル時に行うため、現在利用できるステージングの姿は、動的形状グラフ（`torch.jit.script` の結果）上で形状伝播を行えることだけである。この欠点は、Python コードではなく比較的基本的な IR（Torch Script）をターゲットにするという決定から生じる。ただし、この決定の強力な帰結の一つは、より広範な IR をターゲットとするシステムでは難しい自動バッチ化を、Torch Script 上に明快に実装できることである。

**制約。** Python 言語は大規模であり、AutoGraph はそのすべてをステージングするわけではない。機械学習プログラミングを可能にするサブセットへ重点を置くが、連想データ構造や `try/except` ブロックなど、有用な構文の多くがまだ欠けている。TensorFlow または Lantern の IR に対応する構文がない場合もあるが、対応する IR を増やすにつれ、Python 言語のより多くの部分を正常に変換できるようになると見込んでいる。TensorFlow 構文へ変換されるのは Python 言語の一部だけだが、AutoGraph ではほぼすべての Python 構文を利用でき、変換せずにそのまま呼び出す。これにより、AutoGraph は既存グラフコードの大部分と互換性を持つ。AutoGraph における Python 言語のサポートは、[第 16 節](#section-16)に網羅的に記す。

さらに、AutoGraph が行うデータ依存のステージング判断は、Python の演算子オーバーロードによって、オーバーロードされた演算子内の計算が見えなくなるのとよく似て、ユーザーから見えない。たとえば、ユーザーが条件分岐へ TensorFlow の真偽値ではなく Python の真偽値を誤って渡すと、グラフへステージングされず、性能に影響する可能性がある。現在、この動作を見つけてデバッグするためにユーザーが利用できるツールはほとんどない。本システムを単純に実装した場合より良いエラーメッセージはすでに提供している（[第 13 節](#section-13)を参照）が、さらなる作業が必要である。

Python と IR の型システムの不一致からも、さらなる課題が生じる。たとえば、TensorFlow は null 許容型をサポートしないため、TensorFlow で制御フローをステージングするときは、すべてのコードパスで変数を初期化するよう求め、Python の意味論に追加の制約を課す。同様に、リストのような Python の型はジェネリックなので、要素アクセスには型情報がなく、通常そうであるように IR が強い型付けを持つ場合、追加のユーザー注釈が必要になることがある。こうした注釈を不要にできる、より高度な型推論機構は今後の課題である。

IR への変換が意味論を保つか、さもなければ明示的に失敗することを保証するため、最大限の努力を払っている。しかし、本システムの正しさは、より厳密に扱う必要がある。ランダムコード生成によるファジングシステムを用い、形式的にも実証的にもこの問題へ取り組む予定である。当面は、数百のテストを含む AutoGraph の広範なテストスイートを、正しさの証拠として提示する。さらに AutoGraph は、TensorFlow 2.0 でコードを高速化する既定の方法である `tf.function` に組み込まれているため、TensorFlow コードベース全体を対象とするすべてのテストも受ける。このテストに基づく正しさという考え方は形式的な保証を与えないが、Python 意味論に関するほかの形式解析 [Pol13] と整合していることを付記する。

最後に、AutoGraph は `inspect` や `imp` など、Python のイントロスペクション API とリフレクション API に依存する。これらは大半の利用場面で使用できるが、ソースコード情報が得られない場合など、AutoGraph を利用できない事例もある。

<span id="section-11"></span>

## 11 結論と今後の課題

本稿では、慣用的な Python コードを、TensorFlow グラフやその他のより実験的なバックエンドを含む等価な低水準 IR へ自動的に書き換えるステージドプログラミングシステム、AutoGraph を説明した。AutoGraph は、*命令型*コードと*グラフベース*コードの設計空間で均衡を実現する。実行時オーバーヘッドが大きい完全な命令型と、開発者の認知的負荷が大きい完全なステージド型という二つのプログラミングモデルは、二者択一ではない。SCT を使えば、両者の区別をなくすことができる。本手法は幅広く適用可能だと考えており、新しい用途でより多様な IR をターゲットにする作業を進めている。

AutoGraph の全体は、GitHub の TensorFlow プロジェクトを通じて <https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/autograph> でオープンソース化されている。

## 謝辞

AutoGraph を TensorFlow 2.0 へ統合する際に助力と支援をいただいた Alex Passos 氏と TensorFlow チームの皆様に感謝する。

動的ディスパッチに基づく手法は、Josh Levenberg 氏の先行研究で検討された。

<span id="section-12"></span>

## 12 動的 RNN の実装

以下に、`tf.dynamic_rnn` セルの手書きグラフ実装を示す。

```python
def dynamic_rnn(rnn_cell, input_data,
  initial_state, sequence_len=None):
  input_data = tf.transpose(input_data,
    (1, 0, 2))
  outputs = tf.TensorArray(
    tf.float32, size=0, dynamic_size=True)
  if sequence_length is None:
    max_len = input_data.shape[0]
  else:
    max_len = tf.reduce_max(sequence_len)
  def while_body(i, state, outputs):
    prev_state = state
    output, state = rnn_cell(
      input_data[i], state)
    state = tf.where(
      i < sequence_len,
      state,
      prev_state)
    outputs = outputs.write(i, output)
    return i + 1, state, outputs
  def while_cond(i, state, outputs):
    return i < max_len
  _, state, outputs = tf.while_loop(
    while_cond,
    while_body,
    loop_vars=(tf.constant(0),
      initial_state,
      outputs))
  outputs = outputs.stack()
  outputs = tf.transpose(outputs, (1, 0, 2))
  return outputs, state
```

<span id="section-13"></span>

## 13 エラー処理

AutoGraph には、Python ランタイムが行う通常の構文検証に加えて、明確に異なる三つの実行段階がある：

- 変換
- ステージング（例：TensorFlow グラフの構築）
- 実行時（例：TensorFlow グラフの実行）

後二つの段階は、TensorFlow や PyTorch の JIT モデルのようなプラットフォームが実装する、マルチステージプログラミングモデルの二つのステージに対応付けられる。それぞれの段階でエラー処理の要件は異なるが、主に次の二つの技術を利用する：

- *ソースマップの構築*。SCT のパスを何度か経た後でも、AST の各ノードをユーザーの元の Python コードの行に関連付ける。
- *エラーの書き換え*。TensorFlow コード、とりわけ AutoGraph が生成した TensorFlow コードのスタックトレースには、ユーザーではなく AutoGraph コンパイラシステムが書いたコード行を指すフレームがいくつかある。AutoGraph でコードを生成するときに使う一時ファイルを、ユーザーの元のソースファイルへ再び関連付けることができる。

**変換エラー。** それ自体は正当な Python だが、AutoGraph がサポートしていないコードによって変換エラーが発生することがある。通常、このエラーは AutoGraph の内部コードで生じる。

使いやすさのため、この種のエラーは、原因となった慣用表現が変換対象コードのどこにあるかを示さなければならない。また、開発者がエラーを修正できるだけの情報を、エラーメッセージで提供しなければならない。最後に、内部コードへの参照は通常ユーザーに有用でないため、エラーのスタックトレースに含めるべきではない。

現在は、エラー位置を示すスタックトレース風のメッセージを生成して、この要件を満たしやすくしている。将来は、この種のエラーメッセージをさらに簡潔にする予定である。

**ステージングエラー。** 正常に変換されたコードでもステージングエラーが発生することがあり、通常は、許可されていない、または無効な引数型、形状、ハイパーパラメータ値など、実行時にしか検出できない条件が原因となる。これに対処するため、中間コードの生成元となった元コードのフレームを含む、スタックトレース風のメッセージを生成する予定である。生成した AST の各ノードとユーザーの元のソースコードとの間で保持する AST ソースマップが、これを支える。

エラーメッセージが、生成されたシンボルや生成コード固有の文脈を参照する場合があることも課題である。この欠点への対処は今後の課題となる。

**実行時エラー。** この種類のエラーがいう実行時とは、ステージングされた IR のランタイムを指す。

たとえば、TensorFlow における整数のゼロ除算エラーは次のようになる：

```python
def f(n):
  return tf.constant(10, dtype=tf.int32) / n
```

通常、IR の実行環境には、エラーの発生源をユーザーコードまで追跡する機能が含まれる。しかし、AutoGraph の場合、それは生成コードになる。この問題を解決するため、これらのエラーを捕捉し、発生源を変換前の元コードまでさらに追跡するのに役立つ情報を付加する予定である。TensorFlow 2.0 API に `tf.function` が追加されるのに合わせて、ユーザー体験を改善する予定である。

<span id="section-14"></span>

## 14 有用なユーティリティ

ここまで述べたシステムを構築するため、より広範な Python コミュニティにも有用になると見込まれる、ソースコード変換ツールの大規模なライブラリを作成した。

**コードの容易なクォートとアンクォート。** ユーティリティ関数の一部を次に挙げる：

- `parser.parse_entity(fn_or_class)` は Python のクラスまたは関数を受け取り、対応する AST ノードを、それを格納する `Module` ノードで包んで返す。
- `parser.parse_str(code_string)` は、Python コードの文字列を入力に取る点を除き、`parse_entity` と同一である。文字列には任意の正当な Python コードを含められる。
- `pretty_printer.fmt(ast_node)` は、AST を表す整形表示可能な文字列を返す。
- `compiler.ast_to_source(ast_node)` は、AST を等価な Python コードへ逆解析し、文字列として返す。
- `compiler.ast_to_object(ast_node)` は、AST を等価な Python エンティティへコンパイルし、モジュールとして返す。

たとえば、次のように使う：

```python
node = parse_str('a = b')
print(fmt(node))

# Output:
Module:
| body=[
| | Assign:
| | | targets=[
| | | | Name:
| | | | | id="a"
| | | | | ctx=Store()
| | | | | annotation=None
| | | ]
| | | value=Name:
| | | | id="b"
| | | | ctx=Load()
| | | | annotation=None
| ]
```

これらのユーティリティにより、AST へ小さな変更を加えるのが容易になる。

```python
node = parse_str('a = b')
node.body[0].value.id = 'c'
print(ast_to_source(node))

# Output:
a = c
```

**テンプレートを使ったコード書き換え。** 例：

```python
code_quote = '''
def fn(args):
  body
'''
new_body = textwrap.dedent('''
  a = x
  b = y
  return a + b
''')
node = templates.replace(
  code_quote,
  fn='my_function',
  args=('x', 'y'),
  body=parser.parse_str(new_body).body
)
print(compiler.ast_to_source(node))

# Output:
def my_function(x, y):
  a = x
  b = y
  return a + b
```

この関数は、クォートされたコードテンプレートへ文字列シンボルまたは AST ノードを挿入し、追加の整合性検査を行う。これにより、特に AST を手作業で構築する場合に、複雑なコードブロックを容易に構築できる。

<span id="section-15"></span>

## 15 詳細な例

本文の簡単な例を発展させ、より現実的なアルゴリズムとモデルを実装する際の AutoGraph の有用性を示す。容易に実行できるよう、TensorFlow のベンチマークユーティリティ[+2]を使って実装した。これにより、AutoGraph が生成したコードの性能を、AutoGraph の著者による参照実装と TensorFlow の一部として配布される参照実装の双方と比較することもできる。各例について、予備的な知見を報告する。

本節で言及するすべてのサンプルコードと、論文全体にある例の完全な実行可能コードは、<https://github.com/tensorflow/autograph/examples/sysml2019> にある。

<span id="section-15-1"></span>

### 15.1 ビームサーチ

ビームサーチは、機械翻訳でよく使われるアルゴリズムである。このアルゴリズムは、各遷移で最も確からしいステップを選び、確からしさの低い系列を破棄することもありながら、候補系列を構築する。ビームサーチは、各ステップで複雑な計算と判断を行い、ステップ数には最大系列長による上限があるため、AutoGraph の興味深いユースケースとなる。ビームサーチの最も単純な実装は、すべての候補系列が終了したら抜けるループである。より堅牢な実装では、生存候補系列と終了候補系列を別々に追跡し、どの生存候補も終了候補より高いスコアを得る可能性がなくなればループを抜ける。許容される最大長よりはるかに短い系列を生成することが多いため、ループを抜ける処理はビームサーチの性能に不可欠である。

TensorFlow Eager を使ってビームサーチを実装した。AutoGraph を使うと、同じコードを TensorFlow Eager で実行した場合より、ベンチマークは 2〜3.2 倍高速になる。改善幅は、最大系列長と語彙サイズを変えると変化する。通常、系列が長く語彙が小さいほど、AutoGraph による改善は大きい。系列が長いほどループの反復回数が増えるため、AutoGraph でこれらのループを TensorFlow グラフへ埋め込むことによる相対的な改善が大きくなる。語彙が大きいとベクトル演算と行列演算のコストが増し、全体の所要時間が長くなる。

<span id="section-15-2"></span>

### 15.2 L-BFGS

L-BFGS（Limited-Memory Broyden-Fletcher-Goldfarb-Shannon）アルゴリズムは、機械学習のパラメータ推定によく使われる。本実装は、Yaroslav Bulatov 氏が書いた TensorFlow Eager 実装[+3]に基づく。ベンチマークでは、ほぼ同じコード量で、バッチサイズ 10 のとき AutoGraph は Eager より約 2 倍高速である。

<span id="section-15-3"></span>

### 15.3 モデル非依存メタ学習（MAML）

モデル非依存メタ学習（MAML、[Fin17a]）は、メタ学習のためのアルゴリズムであり、少数ショット学習に特に有効である。本ベンチマークは [Fin17a] の正弦波の例に基づく。[+4]

TensorFlow Eager と AutoGraph の両方に対応するコードを使って MAML ベンチマークを実装した。単一のメタパラメータを学習するとき、AutoGraph で変換したコードは、同一コードを Eager モードで実行した場合より 1.9 倍高速だった。10 個のメタパラメータを学習した場合、AutoGraph で変換したコードは 2.7 倍高速だった。

<span id="section-15-4"></span>

### 15.4 seq2seq

seq2seq（Sequence-to-Sequence）モデル[+5]は、機械翻訳などのタスクに利用できる汎用エンコーダー・デコーダーである。このモデルと、ランダムな入力系列に対するモデルの性能を測るベンチマークを実装した。

このベンチマークを TensorFlow Eager で実装し、その Eager コードを AutoGraph で変換した。AutoGraph で変換したコードは、等価な Eager コードより 1.18〜3.05 倍高速だった。性能向上は語彙サイズによって変わり、AutoGraph は語彙が大きいほど優れた性能を示す。系列長を 64 から 128 まで変えても、性能向上への影響はごく小さかった。さらに、オプションの「teacher forcing」も実装したところ、AutoGraph による改善がほぼ 2 倍になった。teacher forcing によって計算の実行に費やす時間が減り、Eager モードのオーバーヘッドが全体時間に占める割合が大きくなるためである。AutoGraph は、このようなオーバーヘッドを、ここでは TensorFlow が実行するグラフへデータ依存の制御フローを埋め込むことで削減するよう設計されている。

<span id="section-16"></span>

## 16 対応機能

[表 4](#table-04)、[表 5](#table-05)、[表 6](#table-06)に、AutoGraph が現在サポートする Python と TensorFlow の機能を示す。

<span id="table-04"></span>

![AutoGraph における制御フローと演算子のサポート](../../papers/autograph/table-04.png)

**表 4.** AutoGraph の対応機能 [+6] [+7] [+8] [+9] [+10] [+11] [+12] [+13]

<span id="table-05"></span>

![AutoGraph における関数とコレクションのサポート](../../papers/autograph/table-05.png)

**表 5.** AutoGraph の対応機能（続き） [+14] [+15] [+16] [+17] [+18]

<span id="table-06"></span>

![AutoGraph における変数、リテラル、クラス、デコレーター、ジェネレーター、高度な機能のサポート](../../papers/autograph/table-06.png)

**表 6.** AutoGraph の対応機能（続き） [+19] [+20] [+21] [+22] [+23] [+24] [+25] [+26] [+27] [+28]
[+1]: Python Language Reference（<https://docs.python.org/3/reference/>）の 3.3 節を参照。

[+2]: <https://www.tensorflow.org/community/benchmarks>

[+3]: <https://github.com/yaroslavvb/stuff/tree/master/eager_lbfgs>

[+4]: <https://github.com/cbfinn/maml>

[+5]: <https://google.github.io/seq2seq/>

[+6]: 「nest コレクション」とは、`tf.nest` が認識するコレクションである。

[+7]: 例：属性または要素を条件付きで設定する処理が、その属性または要素を常に設定する処理へ変わる可能性がある。TF 2 のリリースで修正する予定である。

[+8]: 例：制御フロー本体の内部で行われる属性と要素の変更は意味論を保つ。関数呼び出し内で行われる変更は、必ずしも意味論を保たない。

[+9]: `tf.while_loop` を使って while ループをステージングする場合、ループ条件は `tf.while_loop` 自体だけが評価する。しかし、`tf.while_loop` が呼ばれる*前*に、ループをステージングするかどうかを判断する必要がある。このため、ループ条件を二度評価することで生じうる Python の副作用を避けるため、事前にはループ条件を評価しない。将来は条件関数を二度評価し、この意味論を明確に文書化する予定である。

[+10]: 現在、TensorFlow には例外を捕捉する機能がない。

[+11]: 近く `yield` を変換なしでサポートする予定である。

[+12]: ただし、通常 `Tensor` オブジェクトはすべての算術演算子をオーバーロードし、式は TF 演算へステージングされる点に注意されたい。

[+13]: たとえば、Python の遅延真偽値評価の意味論と整合させるため、`x and y` は `tf.cond(x, lambda: y, lambda: x)` へ変換される。

[+14]: 現在、ホワイトリストには TF モジュールが含まれる。

[+15]: すなわち、`to_graph` または `tf.function` に直接渡されたユーザー関数は常に変換される。

[+16]: すなわち、第 1 引数に `self` を取る関数である。

[+17]: すべての Python 組み込み関数に、対応する TF 演算があるわけではない。

[+18]: 対応する TF 演算が追加されるのに合わせて、サポートを追加する予定である。

[+19]: 長期的には Python の意味論へ完全に従い、未定義変数へアクセスしたとき実行時例外を発生させる予定である。

[+20]: 近くサポートする予定である。

[+21]: 近くサポートする予定である。

[+22]: 多くの TF 演算は、特定の値を `Tensor` へ自動ボックス化する点に注意されたい。

[+23]: すなわち、`to_graph` または `tf.function` に直接渡されたユーザークラスは常に変換される。

[+24]: 「ユーザー関数」を参照。たとえば Keras Model クラスのサブクラスは、Model クラスから継承したメソッドではなく、サブクラスで定義されたメソッドだけを変換する。

[+25]: 例：`functools.lru_cache` はサポートされない。`functools.wraps` はサポートされるが、変換されない。

[+26]: 変換せずにジェネレーターを使えるようにする予定である。

[+27]: `pdb` 呼び出しは生成コードへ挿入され、グラフ構築時のステージングで有効になる。

[+28]: `getsource` など、一部の `inspect` API は正しく動作するが、広範なテストは行っていない。
