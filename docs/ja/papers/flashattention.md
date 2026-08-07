---
title: 'FlashAttention'
createTime: 2026/08/05 00:15:31
permalink: /ja/papers/flashattention/
---

> [Tri Dao](https://tridao.me/), [Daniel Y. Fu](https://danfu.org/), [Stefano Ermon](https://cs.stanford.edu/~ermon/), [Atri Rudra](http://www.cse.buffalo.edu/~atri/), and [Christopher Ré](http://cs.stanford.edu/people/chrismre/). arXiv 初回投稿日: May 27, 2022; 現行版は v2. [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135). [原 PDF](/paper/flashattention.pdf). [TeX ソース](https://export.arxiv.org/e-print/2205.14135). 正確な印刷レイアウトと参考文献については原 PDF を正本とする.

## 要約

トランスフォーマーは長いシーケンスに対して遅く、メモリを多く消費します。これは自己注意機構の時間とメモリの複雑性がシーケンス長の二乗に比例するためです。近似注意機構は、計算複雑性を減らすためにモデルの品質と引き換えにこの問題に対処しようと試みてきましたが、実際の処理時間の短縮には必ずしもつながりません。私たちは、欠けている原理は注意アルゴリズムをIO対応にすることであり、GPUメモリ階層間の読み書きを考慮することだと考えています。私たちはFlashAttentionを提案します。これはIO対応の正確な注意アルゴリズムであり、GPUの高帯域メモリ（HBM）とGPU内蔵のSRAM間でのメモリ読み書き回数を減らすためにタイル処理を活用します。FlashAttentionのIO複雑性を解析した結果、標準的な注意よりもHBMアクセスが少なく、SRAMサイズの範囲で最適であることを示しました。また、FlashAttentionをブロックスパース注意に拡張し、既存の近似注意方法よりも高速な近似注意アルゴリズムを実現しました。FlashAttentionは既存のベースラインよりも速くトランスフォーマーを訓練します：BERT-large（シーケンス長512）でMLPerf 1.1の訓練速度記録に対してエンドツーエンドで15%の時間短縮、GPT-2（シーケンス長1K）で3$\times$の速度向上、ロングレンジアリーナ（シーケンス長1K-4K）で2.4$\times$の速度向上です。FlashAttentionおよびブロックスパースFlashAttentionは、トランスフォーマーでより長い文脈を可能にし、より高品質なモデルを実現します（GPT-2でのパープレキシティ0.7改善、長文分類での6.4ポイント向上）および全く新しい能力をもたらします：Path-Xチャレンジ（シーケンス長16K, 精度61.4%）およびPath-256（シーケンス長64K, 精度63.1%）で、初めて偶然以上の性能を達成したトランスフォーマーです。

## 1 はじめに

トランスフォーマーモデル [Advand17] は、自然言語処理や画像分類などのアプリケーションで最も広く使われるアーキテクチャとして登場しました。トランスフォーマーはより大きく [Advanb01]、より深く [Wang22] なっていますが、長いコンテキストに対応させることは依然として難しいです [Repree20]。なぜなら、その中心にある自己注意モジュールはシーケンス長に対して二次の時間とメモリの複雑性を持つからです。重要な疑問は、注意機構をより高速かつメモリ効率良くすることが、長いシーケンスに対するトランスフォーマーモデルの実行時間およびメモリの課題解決に役立つかどうかです。

多くの近似注意メソッドは、注意機構の計算およびメモリの要求を減らすことを目的としています。これらの方法は、スパース近似 [ICML20, Roy21] から低ランク近似 [Wang20, Kathar20, Chorom20]、そしてその組み合わせ [Beltag20, Systeo20, Chen21] まで多岐にわたります。これらの方法は、計算要求をシーケンス長に対して線形またはほぼ線形に減らしますが、多くの場合、標準的な注意機構に対して実時間の高速化を示さず、広く採用されていません。その主な理由の一つは、これらが FLOP の削減（必ずしも実時間の短縮と相関しない場合がある）に焦点を当て、メモリアクセス（IO）によるオーバーヘッドを無視しがちであることです。

<span id="figure-01"></span>

![キャプションを参照](../../papers/flashattention/figure-01.png)

**図1.** 左：FlashAttentionはタイル処理を使用して、大きな$N\times N$注意行列（点線の枠）が（比較的）遅いGPU HBM上で具体化されるのを防ぐ。外側のループ（赤い矢印）では、FlashAttentionは$\mathbf{K}$および$\mathbf{V}$行列のブロックをループ処理し、それらを高速なオンチップSRAMにロードする。各ブロック内で、FlashAttentionは$\mathbf{Q}$行列のブロックをループ処理（青い矢印）、SRAMにロードし、注意計算の出力をHBMに書き戻す。右：GPT-2上での注意処理のPyTorch実装に対する速度向上。FlashAttentionは大きな$N\times N$注意行列をHBMに読み書きしないため、注意計算で7.6$\times$の速度向上が得られる。

本論文では、注意アルゴリズムをIO認識[Vitter88]にするという欠落する原則、すなわち高速・遅いメモリレベル（例えば高速GPUオンチップSRAMと比較的遅いGPU高帯域幅メモリ（HBM [Jiaa18]）の読み取りと書き込みを慎重に考慮することである[図1](#figure-01)左）を主張します。現代のGPUでは、計算速度がメモリ速度[GPUd17, GPU20, GPU22]を上回っており、トランスフォーマーのほとんどの操作はメモリアクセス[Ivanov21]によってボトルネックされています。IO認識アルゴリズムは、データベースの結合[Ramakr03]、画像処理[Notice13]、数値線形代数[Willia09, Approa03] [Blackf02]など、データの読み書きが実行時間の大部分を占める場合、同様のメモリバウンド操作において重要な役割を果たしてきました。しかし、PyTorchやTensorflowのような一般的なPythonのディープラーニングインターフェースでは、メモリアクセスの細かな制御はできません。

我々は、はるかに少ないメモリアクセスで正確なアテンションを計算できる新しいアテンションアルゴリズム、FlashAttentionを提案します。我々の主な目標は、アテンション行列をHBMから読み書きすることを避けることです。これには、（i） 入力全体にアクセスせずにソフトマックスの集約を計算すること、（ii） バックワードパスのために大きな中間アテンション行列を保存しないこと、が必要です。我々はこれらの課題に対処するために、二つの確立された技術を適用します。（i） アテンション計算を再構築し、入力をブロックに分割し、入力ブロックを複数回走査することで、ソフトマックスの集約を段階的に行います（タイル化とも呼ばれます）。（ii） フォワードパスからソフトマックスの正規化係数を保存し、バックワードパスでオンチップで迅速にアテンションを再計算できるようにします。これにより、中間アテンション行列をHBMから読み込む標準的な方法よりも高速になります。我々は、メモリアクセスを細かく制御するためにCUDAでFlashAttentionを実装し、すべてのアテンション操作を1つのGPUカーネルに統合しました。再計算によるFLOPsの増加にもかかわらず、我々のアルゴリズムは、GPT-2 [Radfor19]で最大7.6倍高速に動作し（[図 1](#figure-01) 右）、HBMアクセスの大幅削減により、標準アテンションよりもシーケンス長に比例した少ないメモリを使用します。

私たちは FlashAttention の IO 複雑度 [Vitter88] を分析し、ヘッド次元を $d$、SRAM のサイズを $M$ とした場合に $O(N^{2}d^{2}M^{-1})$ の HBM アクセスが必要であることを証明しました。これは標準的な注意機構の $\Omega(\mathrm{Nd}+N^{2})$ と比較されます。$d$ と $M$ の典型的な値において、FlashAttention は標準的な注意機構と比べてはるかに少ない HBM アクセスで済みます（[図 2](#figure-02) に示されるように最大で $\times$ 倍少なくなります）。さらに、下限も示しており、どの正確な注意アルゴリズムも SRAM サイズ全体で HBM アクセスの数を漸近的に改善することはできないことを示しています。

また、FlashAttention は、メモリアクセスのオーバーヘッドの問題を克服することにより、近似注意アルゴリズムの潜在能力を実現するための有用な基本要素として機能することを示します。概念実証として、ブロックスパース FlashAttention を実装します。これはスパース注意アルゴリズムで、FlashAttention よりも 2-4$\times$ 倍高速で、シーケンス長 64k までスケール可能です。ブロックスパース FlashAttention はスパース比に比例して FlashAttention より優れた IO 複雑度を持つことを証明します。他の操作（マルチ GPU 上の注意、カーネル回帰、ブロックスパース行列乗算）へのさらなる拡張については [セクション 5](#S5) で議論します。我々は、この基本要素の構築を容易にするために FlashAttention をオープンソース化します。[+1]

FlashAttentionがモデルのトレーニングを高速化し、より長い文脈をモデル化することでモデルの品質を向上させることを実証的に検証しました。また、従来のアテンション実装と比較して、FlashAttentionおよびブロックスパースFlashAttentionの実行時間とメモリ使用量をベンチマークしました。

-   •

より高速なモデルトレーニング。FlashAttentionは、Transformerモデルをウォールクロック時間でより速くトレーニングします。BERT-large（シーケンス長512）のトレーニングをMLPerf 1.1のトレーニング速度記録より15％速く、GPT2（シーケンス長1K）をHuggingFaceおよびMegatron-LMのベースライン実装より3倍速く、long-range arena（シーケンス長1K-4K）ではベースラインより2.4倍速くトレーニングしました。

-   •

より高品質なモデル。FlashAttentionはTransformerをより長いシーケンスに拡張できるため、その品質を向上させ、新しい機能を可能にします。GPT-2ではパープレキシティが0.7改善され、長文分類の長いシーケンスのモデル化では6.4ポイントの向上が観察されました。FlashAttentionは、シーケンス長16Kを使用することでのみ、Path-Xチャレンジで偶然より良いパフォーマンスを達成できる初のTransformerを可能にします。ブロックスパースFlashAttentionは、Transformerをさらに長いシーケンス（64K）にスケーリングできるようにし、Path-256で偶然より良いパフォーマンスを達成できる初のモデルを実現します。

-   •

注意力のベンチマーク。FlashAttentionは、128から2Kまでの一般的なシーケンス長で標準的な注意力実装より最大3$\times$速く、64Kまでスケールします。シーケンス長が512までの場合、FlashAttentionは既存のどの注意力手法よりも高速でメモリ効率も良いですが、シーケンス長が1Kを超えると、一部の近似注意力手法（例：Linformer）がより速くなることがあります。一方、ブロックスパースFlashAttentionは、私たちが知る限り既存のすべての近似注意力手法よりも高速です。

## 2 背景

現代のハードウェア（GPU）上での一般的なディープラーニング操作の性能特性についていくつか背景を提供します。また、注意力の標準的な実装についても説明します。

### 2.1 ハードウェア性能

ここではGPUに焦点を当てます。他のハードウェアアクセラレータでの性能も同様です[Archia17, Jiaa19]。

GPUメモリ階層。GPUメモリ階層（[図1](#figure-01) 左側）は、異なるサイズと速度の複数のメモリで構成されており、サイズが小さいメモリほど高速です。例として、A100 GPUは40〜80GBの高帯域メモリ（HBM）を持ち、帯域幅は1.5〜2.0TB/sであり、108個のストリーミングマルチプロセッサごとに192KBのオンチップSRAMを持ち、帯域幅は約19TB/sと推定されます [Jiaa18, Sandt21]。オンチップSRAMはHBMよりも1桁高速ですが、サイズははるかに小さいです。計算速度がメモリ速度に対して速くなるにつれて [GPUd17, GPU20, GPU22]、操作はますますメモリ（HBM）アクセスによってボトルネックになります。したがって、高速なSRAMを活用することがより重要になります。

実行モデル。GPUは大量のスレッドを持ち、操作（カーネルと呼ばれる）を実行します。各カーネルはHBMからレジスタとSRAMに入力を読み込み、計算を行い、その後出力をHBMに書き込みます。

性能特性。計算とメモリアクセスのバランスに応じて、操作は計算バウンドまたはメモリバウンドに分類されます。これは一般的に*算術強度* [Willia09]で測定され、メモリアクセス1バイトあたりの算術演算の数です。

1. 1.

計算バウンド：操作にかかる時間は算術演算の数によって決まり、HBMへのアクセス時間ははるかに短いです。典型的な例は大規模な内部次元を持つ行列乗算や、多数のチャネルを持つ畳み込みです。

2. 2.

メモリバウンド： 操作にかかる時間はメモリアクセスの回数によって決まり、計算に費やされる時間ははるかに短い。例としては、ほとんどの他の操作が含まれる： 要素ごとの操作（例： アクティベーション、ドロップアウト）、および縮約（例： 合計、ソフトマックス、バッチ正規化、レイヤー正規化）。

カーネルフュージョン。メモリバウンド操作を高速化する最も一般的な方法はカーネルフュージョンである： 同じ入力に対して複数の操作が適用される場合、各操作のために何度も読み込むのではなく、入力をHBMから一度だけロードできる。コンパイラは多くの要素ごとの操作を自動的にフュージョンできる[Refa20, Advane19, Compil20]。しかし、モデル訓練の文脈では、中間値はバックワードパスのためにHBMに書き込む必要があり、素朴なカーネルフュージョンの効果を減少させる。

### 2.2 標準的なアテンションの実装

入力シーケンスが与えられた場合$\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}$ ここで$N$はシーケンス長、$d$はヘッド次元であり、アテンション出力$\mathbf{O}\in\mathbb{R}^{N\times d}$を計算したい：

$$
\mathbf{S}=\mathbf{Q}\mathbf{K}^{\top}\in\mathbb{R}^{N\times N},\quad\mathbf{P}=\mathrm{softmax}(\mathbf{S})\in\mathbb{R}^{N\times N},\quad\mathbf{O}=\mathbf{P}\mathbf{V}\in\mathbb{R}^{N\times d},
$$

ここで$\mathrm{softmax}$は行ごとに適用される。

標準的なアテンション実装は、行列 $\mathbf{S}$ および $\mathbf{P}$ を HBM にマテリアライズし、これに $O(N^{2})$ のメモリを消費します。多くの場合 $N\gg d$ （例：GPT2 では $N=1024$ および $d=64$）。標準アテンション実装については [アルゴリズム](#alg0) で説明します。一部またはほとんどの操作がメモリバウンド（例えばソフトマックス）であるため、メモリアクセスの多さが実時間の遅延につながります。

この問題は、$\mathbf{S}$ に適用されるマスキングや $\mathbf{P}$ に適用されるドロップアウトなど、アテンション行列に適用される他の要素ごとの操作によって悪化します。その結果、マスキングとソフトマックスの融合 [Shoeyb19] のように、いくつかの要素ごとの操作を統合する試みが多くなされています。

[セクション 3.2](#S3.SS2) では、標準アテンション実装がシーケンス長 $N$ に対して HBM アクセスを二乗的に行うことを示します。また、標準アテンションと我々の手法（FlashAttention）の FLOPs 数および HBM アクセス数を比較します。

<span id="alg0"></span>

**アルゴリズム 0: 標準アテンション実装**

- **入力:** 行列 $\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}$ を HBM に配置する。
- <span id="alg0.l1"></span> HBMからブロック単位で$\mathbf{Q},\mathbf{K}$をロードし、$\mathbf{S}=\mathbf{Q}\mathbf{K}^{\top}$を計算して、$\mathbf{S}$をHBMに書き込む。
- <span id="alg0.l2"></span> HBMから$\mathbf{S}$を読み込み、$\mathbf{P}=\mathrm{softmax}(\mathbf{S})$を計算して、$\mathbf{P}$をHBMに書き込む。
- <span id="alg0.l3"></span> HBMから$\mathbf{P}$と$\mathbf{V}$をブロック単位でロードし、$\mathbf{O}=\mathbf{P}\mathbf{V}$を計算して、$\mathbf{O}$をHBMに書き込む。
- **返却:** $\mathbf{O}$を返す。

## 3 FlashAttention： アルゴリズム、解析、拡張

我々は、より少ないHBMの読み書きで正確なアテンションを計算し、バックワードパスのための大規模な中間行列を保存せずに済む方法を示す。これにより、メモリ効率が高く、実時間でより高速なアテンションアルゴリズムが得られる。我々はそのIO計算量を解析し、標準的なアテンションと比較して、我々の方法がはるかに少ないHBMアクセスで済むことを示す。さらに、ブロックスパースアテンションに対応するように拡張することで、FlashAttentionが有用なプリミティブとして機能することも示す。

説明の簡便さのためにここではフォワードパスに焦点を当てる； バックワードの詳細は[付録 B](#A2)に記載されている。

### 3.1 タイリングと再計算による効率的なアテンションアルゴリズム

HBM に与えられた入力 $\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}$ を基に、アテンション出力 $\mathbf{O}\in\mathbb{R}^{N\times d}$ を計算し、それを HBM に書き込むことを目指します。我々の目標は、HBM へのアクセス量を削減すること（$N$ に対して二次以下にすること）です。

正確なアテンションをサブ二次的な HBM アクセスで計算する技術的課題を克服するために、2 つの確立された手法（タイル処理、再計算）を適用します。これについては [アルゴリズム 1](#alg1) で説明しています。主なアイデアは、入力 $\mathbf{Q},\mathbf{K},\mathbf{V}$ をブロックに分割し、それを遅い HBM から高速 SRAM にロードしてから、それらのブロックに関してアテンション出力を計算することです。各ブロックの出力を適切な正規化係数でスケーリングしてから合算することで、最終的に正しい結果を得ることができます。

タイル処理。ブロックごとにアテンションを計算します。Softmax は $\mathbf{K}$ の列同士を結合するため、大きな softmax をスケーリング [Gimels18, ICML20, Staats21] で分解します。数値の安定性のために、ベクトル $x\in\mathbb{R}^{B}$ の softmax は次のように計算されます：

$$
m(x):=\max_{i}\ \ x_{i},\quad f(x):=\begin{bmatrix}e^{x_{1}-m(x)}&\ldots&e^{x_{B}-m(x)}\end{bmatrix},\quad\ell(x):=\sum_{i}f(x)_{i},\quad\mathrm{softmax}(x):=\frac{f(x)}{\ell(x)}.
$$

ベクトル $x^{(1)},x^{(2)}\in\mathbb{R}^{B}$ に対して、連結された $x=\begin{bmatrix}x^{(1)}\ x^{(2)}\end{bmatrix}\in\mathbb{R}^{2B}$ の softmax を次のように分解することができます：

$$
m(x)=m(\begin{bmatrix}x^{(1)}\ x^{(2)}\end{bmatrix})=\max(m(x^{(1)}),m(x^{(2)})),\quad f(x)=\begin{bmatrix}e^{m(x^{(1)})-m(x)}f(x^{(1)})&e^{m(x^{(2)})-m(x)}f(x^{(2)})\end{bmatrix},
$$

$$
\ell(x)=\ell(\begin{bmatrix}x^{(1)}\ x^{(2)}\end{bmatrix})=e^{m(x^{(1)})-m(x)}\ell(x^{(1)})+e^{m(x^{(2)})-m(x)}\ell(x^{(2)}),\quad\mathrm{softmax}(x)=\frac{f(x)}{\ell(x)}.
$$

したがって、いくつかの追加統計情報を追跡すれば（$m(x),\ell(x)$）、ソフトマックスを一度にブロックごとに計算することができます。[+2] そのため、入力をブロックに分割します（$\mathbf{Q},\mathbf{K},\mathbf{V}$ [アルゴリズム1](#alg1) の行 [3](#alg1.l3)）、追加統計情報と共にソフトマックス値を計算し（[アルゴリズム1](#alg1) の行 [10](#alg1.l10)）、結果を結合します（[アルゴリズム1](#alg1) の行 [12](#alg1.l12)）。

再計算。我々の目標の一つは、逆伝播のために$O(N^{2})$ 中間値を保存しないことです。逆伝播は通常、$\mathbf{S},\mathbf{P}\in\mathbb{R}^{N\times N}$ 行列を用いて $\mathbf{Q},\mathbf{K},\mathbf{V}$ に関する勾配を計算する必要があります。しかし、出力 $\mathbf{O}$ とソフトマックス正規化の統計情報 $(m,\ell)$ を保存することにより、SRAM 内の $\mathbf{Q},\mathbf{K},\mathbf{V}$ ブロックから、逆伝播で簡単に注意行列 $\mathbf{S}$ および $\mathbf{P}$ を再計算することができます。これは選択的勾配チェックポイント [SIAM08, Chen16] の一形態と見なすことができます。勾配チェックポイントは必要な最大メモリ量を削減するために提案されていますが [Staats21]、我々の知る限りでは、すべての実装は速度とメモリをトレードオフする必要があります。それに対して、FLOPs が増えても、我々の再計算は HBM アクセス量が減るため逆伝播を高速化します（[図 2](#figure-02)）。逆伝播の完全な説明は [付録 B](#A2) にあります。

実装の詳細： カーネルフュージョン。タイル化により、HBMから入力を読み込み、すべての計算ステップ（行列乗算、ソフトマックス、必要に応じてマスキングとドロップアウト、行列乗算）を実行し、結果をHBMに書き戻すというアルゴリズムを1つのCUDAカーネルで実装することが可能になります（マスキングとドロップアウトは[付録 B](#A2)に記載されています）。これにより、HBMへの入力と出力の読み書きを繰り返すことを回避できます。

<span id="alg1"></span>

**アルゴリズム 1: FlashAttention**

- **入力:** HBMにある行列$\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}$、オンチップSRAMのサイズ$M$。
- ブロックサイズ$B_{c}=\left\lceil\frac{M}{4d}\right\rceil,B_{r}=\min\left(\left\lceil\frac{M}{4d}\right\rceil,d\right)$を設定する。
- <span id="alg1.l2"></span> HBM内で$\mathbf{O}=(0)_{N\times d}\in\mathbb{R}^{N\times d},\ell=(0)_{N}\in\mathbb{R}^{N},m=(-\infty)_{N}\in\mathbb{R}^{N}$を初期化する。
- <span id="alg1.l3"></span> $\mathbf{Q}$を各$B_{r}\times d$サイズの$\mathbf{Q}_{1},\dots,\mathbf{Q}_{T_{r}}$ブロックに分割し、$\mathbf{K},\mathbf{V}$を各$B_{c}\times d$サイズの$T_{c}=\left\lceil\frac{N}{B_{c}}\right\rceil$ブロックおよび$\mathbf{K}_{1},\dots,\mathbf{K}_{T_{c}}$と$\mathbf{V}_{1},\dots,\mathbf{V}_{T_{c}}$に分割する。
- $\mathbf{O}$ を $T_{r}$ ブロック $\mathbf{O}_{i},\dots,\mathbf{O}_{T_{r}}$ に分割し、それぞれのサイズは $B_{r}\times d$ とし、$\ell$ を $T_{r}$ ブロック $\ell_{i},\dots,\ell_{T_{r}}$ に分割し、それぞれのサイズは $B_{r}$ とし、$m$ を $T_{r}$ ブロック $m_{1},\dots,m_{T_{r}}$ に分割し、それぞれのサイズは $B_{r}$ とする。
- <span id="alg1.l5"></span> **$1\leq j\leq T_{c}$ に対して繰り返す:**
  - <span id="alg1.l6"></span> $\mathbf{K}_{j},\mathbf{V}_{j}$ を HBM からオンチップ SRAM にロード。
  - **$1\leq i\leq T_{r}$ に対して繰り返す:**
    - <span id="alg1.l8"></span> $\mathbf{Q}_{i},\mathbf{O}_{i},\ell_{i},m_{i}$ を HBM からオンチップ SRAM にロード。
    - <span id="alg1.l9"></span> チップ上で、$\mathbf{S}_{\mathrm{ij}}=\mathbf{Q}_{i}\mathbf{K}_{j}^\top\in\mathbb{R}^{B_{r}\times B_{c}}$ を計算します。
    - <span id="alg1.l10"></span> チップ上で、$\tilde{m}_{\mathrm{ij}}=\mathrm{rowmax}(\mathbf{S}_{\mathrm{ij}})\in\mathbb{R}^{B_{r}}$、$\tilde{\mathbf{P}}_{\mathrm{ij}}=\exp(\mathbf{S}_{\mathrm{ij}}-\tilde{m}_{\mathrm{ij}})\in\mathbb{R}^{B_{r}\times B_{c}}$（要素ごと）、$\tilde{\ell}_{\mathrm{ij}}=\mathrm{rowsum}(\tilde{\mathbf{P}}_{\mathrm{ij}})\in\mathbb{R}^{B_{r}}$ を計算します。
    - チップ上で、$m_{i}^{\mathrm{new}}=\max(m_{i},\tilde{m}_{\mathrm{ij}})\in\mathbb{R}^{B_{r}}$、$\ell_{i}^{\mathrm{new}}=e^{m_{i}-m_{i}^{\mathrm{new}}}\ell_{i}+e^{\tilde{m}_{\mathrm{ij}}-m_{i}^{\mathrm{new}}}\tilde{\ell}_{\mathrm{ij}}\in\mathbb{R}^{B_{r}}$ を計算します。
    - <span id="alg1.l12"></span> $\mathbf{O}_{i}\leftarrow\mathrm{diag}(\ell_{i}^{\mathrm{new}})^{-1}(\mathrm{diag}(\ell_{i})e^{m_{i}-m_{i}^{\mathrm{new}}}\mathbf{O}_{i}+e^{\tilde{m}_{\mathrm{ij}}-m_{i}^{\mathrm{new}}}\tilde{\mathbf{P}}_{\mathrm{ij}}\mathbf{V}_{j})$ を HBM に書き込む。
    - $\ell_{i}\leftarrow\ell_{i}^{\mathrm{new}}$、$m_{i}\leftarrow m_{i}^{\mathrm{new}}$ を HBM に書き込む。
- **返却:** $\mathbf{O}$ を返す。

FlashAttention の正確性、実行時間、メモリ要件を示す（証明は [付録 C](#A3) に記載）。

###### 定理 1.

[アルゴリズム 1](#alg1) は $\mathbf{O}=\mathrm{softmax}(\mathbf{Q}\mathbf{K}^{\top})\mathbf{V}$ を $O(N^{2}d)$ FLOPs で返し、入力と出力に加えて $O(N)$ の追加メモリを必要とする。

### 3.2 分析： FlashAttention の IO 複雑度

我々はFlashAttentionのIO複雑性を分析し、標準的なアテンションに比べてHBMアクセスが大幅に削減されることを示します。また、下限も提供し、すべてのSRAMサイズに対して、いかなる正確なアテンションアルゴリズムもHBMアクセスを漸近的に改善できないことを証明します。証明は[付録C](#A3)にあります。

###### 定理 2.

$N$ をシーケンス長、$d$ をヘッド次元、$M$ を $d\leq M\leq \mathrm{Nd}$ の SRAM サイズとする。標準アテンション（[アルゴリズム](#alg0)）は $\Theta(\mathrm{Nd}+N^{2})$ の HBM アクセスを必要とするのに対し、FlashAttention（［アルゴリズム1］（#alg1 "In 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention： Algorithm, Analysis, and Extensions ‣ FlashAttention： Fast and Memory-Efficient Exact Attention with IO-Awareness" ））は $\Theta(N^{2}d^{2}M^{-1})$ の HBM アクセスを必要とする。

$d$（64-128）および$M$（約100KB）の典型的な値に対して、$d^{2}$ は$M$よりもはるかに小さいため、FlashAttentionは標準実装に比べてHBMアクセスが何倍も少なくて済みます。これにより、[セクション 4.3](#S4.SS3)で検証したように、実行速度の向上とメモリ使用量の削減の両方につながります。

証明の主なアイデアは、$M$ の SRAM サイズが与えられた場合、サイズ $\Theta(M)$ の $\mathbf{K},\mathbf{V}$ のブロックをロードできるということです（[アルゴリズム 1](#alg1) の [6 行目](#alg1.l6)）。各 $\mathbf{K}$ および $\mathbf{V}$ のブロックに対して、$\mathbf{Q}$ のすべてのブロック上で反復処理を行い（[アルゴリズム 1](#alg1) の [8 行目](#alg1.l8)）、中間値を計算し、その結果 $\mathbf{Q}$ に対して $\Theta(\mathrm{NdM}^{-1})$ 回処理を行います。各処理では $\Theta(\mathrm{Nd})$ の要素をロードし、$\Theta(N^{2}d^{2}M^{-1})$ の HBM アクセスに相当します。同様に、標準アテンションの逆方向処理は $\Theta(\mathrm{Nd}+N^{2})$ の HBM アクセスを必要とし、FlashAttention の逆方向処理は $\Theta(N^{2}d^{2}M^{-1})$ の HBM アクセスを必要とすることを証明します（[付録 B](#A2)）。

我々は下界を証明する：正確なアテンションを計算する際に、$M$（SRAMのサイズ）のすべての値に対して、HBMアクセスの数を漸近的に改善することはできないことを示す。

###### 命題 3.

$N$をシーケンス長、$d$をヘッド次元、$M$を$d\leq M\leq \mathrm{Nd}$でのSRAMのサイズとする。$[d,\mathrm{Nd}]$の範囲内のすべての$M$について、$o(N^{2}d^{2}M^{-1})$のHBMアクセスで正確なアテンションを計算するアルゴリズムは存在しない。

証明は、$M=\Theta(\mathrm{Nd})$に対して任意のアルゴリズムが$\Omega(N^{2}d^{2}M^{-1})=\Omega(\mathrm{Nd})$のHBMアクセスを行わなければならないという事実に基づく。この種類の下限は、$M$のサブレンジにおいてストリーミングアルゴリズム文献で一般的である [Citese04]。我々は、$M$の観点でパラメータ化された複雑性 [Grohe06]の下限を証明することを、今後の興味深い課題として残す。

私たちは、HBMアクセスの回数が注意機構の実行時間を決定する主な要因であることを検証します。[図2](#figure-02)（左）では、FlashAttentionは（逆伝播での再計算のために）標準の注意機構よりもFLOP数が多いにもかかわらず、HBMアクセスがはるかに少ないため、実行時間が大幅に短くなることがわかります。[図2](#figure-02)（中央）では、FlashAttentionのブロックサイズ$B_{c}$を変更し、異なるHBMアクセス量を発生させ、順方向パスの実行時間を測定します。ブロックサイズが大きくなると、HBMアクセスの回数が減少し（入力に対するパスの回数が減るため）、実行時間も短くなります。十分に大きなブロックサイズ（256を超える場合）では、実行時間は他の要因（例えば算術演算）によってボトルネックとなります。さらに、大きなブロックサイズは小さいSRAMサイズに収まりません。

<span id="figure-02"></span>

![キャプションを参照](../../papers/flashattention/figure-02.png)

**図2.** 左： A100 GPU上でのGPT-2 medium（シーケンス長1024、ヘッド次元64、ヘッド数16、バッチサイズ64）の標準注意機構およびFlashAttentionの順方向・逆方向実行時間。HBMアクセスが実行時間に影響する主要な要素です。中央： A100 GPU上でのFlashAttention（シーケンス長1024、ヘッド次元64、ヘッド数16、バッチサイズ64）の順方向実行時間。HBMアクセスが少ないほど実行時間が速くなりますが、ある点までです。右： ブロックスパースFlashAttention（シーケンス長4K）の実行時間は、スパース性に比例してFlashAttentionよりも速いです。

### 3.3 拡張：ブロックスパースFlashAttention

我々はFlashAttentionを近似注意に拡張する： IO複雑度がスパース性に比例してFlashAttentionより小さいブロックスパースFlashAttentionを提案する。

入力$\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}$とマスク行列$\tilde{\mathbf{M}}\in\{0,1\}^{N\times N}$が与えられた場合、次を計算したい：

$$
\mathbf{S}=\mathbf{Q}\mathbf{K}^{\top}\in\mathbb{R}^{N\times N},\quad\mathbf{P}=\mathrm{softmax}(\mathbf{S}\odot\vmathbb{1}_{\tilde{\mathbf{M}}})\in\mathbb{R}^{N\times N},\quad\mathbf{O}=\mathbf{P}\mathbf{V}\in\mathbb{R}^{N\times d},
$$

ここで $(\mathbf{S}\odot\vmathbb{1}_{\tilde{\mathbf{M}}})_{\mathrm{kl}}=\mathbf{S}_{\mathrm{kl}}$ は $\tilde{\mathbf{M}}_{\mathrm{kl}}=1$ の場合、$-\infty$ は $\mathbf{M}_{\mathrm{kl}}=0$ の場合とする。我々は $\tilde{\mathbf{M}}$ がブロック形式を持つことを要求する：いくつかのブロックサイズ $B_{r},B_{c}$ に対して、すべての $k,l$ について、$\tilde{\mathbf{M}}_{k,l}=\mathbf{M}_{\mathrm{ij}}$ が $i=\lfloor k/B_{r}\rfloor,j=\lfloor l/B_{c}\rfloor$ であるようにして、いくつかの $\mathbf{M}\in\{0,1\}^{N/B_{r}\times N/B_{c}}$ が存在する。

事前に定義されたブロックスパースマスク $\mathbf{M}\in\{0,1\}^{N/B_{r}\times N/B_{c}}$ が与えられた場合、[アルゴリズム 1](#alg1) を簡単に適応させて、注意行列の非ゼロブロックのみを計算することができます。このアルゴリズムは [アルゴリズム 1](#alg1) と同一で、ゼロブロックを飛ばすだけです。[アルゴリズム 5](#alg5) に記載されたアルゴリズムの説明を [付録 B](#A2) に再掲します。

また、ブロックスパース FlashAttention の IO 複雑性も解析します。

###### 命題 4.

ここで、$N$ をシーケンス長、$d$ をヘッド次元、$M$ を $d\leq M\leq \mathrm{Nd}$ を持つ SRAM のサイズとします。ブロックスパース FlashAttention（[アルゴリズム 5](#alg5)）は、$\Theta(\mathrm{Nd}+N^{2}d^{2}M^{-1}s)$ の HBM アクセスを必要とします。ここで、$s$ はブロックスパースマスクにおける非ゼロブロックの割合です。

ブロックスパースを適用すると、IO 複雑性の大きな項に対してスパース性による直接的な改善が得られることがわかります。大きなシーケンス長 $N$ では、$s$ はよく $N^{-1/2}$ [Child19] または $N^{-1}\log N$ [Systeo20, Beltag20, Daoa22] に設定され、$\Theta(N\sqrt{N})$ または $\Theta(N\log N)$ の IO 複雑性が得られます。下流の実験では、固定されたバタフライスパースパターン [Daoa22] を使用します。これは任意のスパース性 [Dao20] を近似できることが示されています。

[図 2](#figure-02) （右） では、スパース性が増加するにつれて、ブロックスパース FlashAttention の実行時間が比例して改善することを確認します。LRA ベンチマークでは、ブロックスパース FlashAttention は 2.8$\times$ のスピードアップを達成し、標準の Attention （[セクション 4](#S4)） と同等の性能を発揮します。

## 4 実験

私たちは、Transformerモデルの訓練にFlashAttentionを使用することの影響を評価します。訓練時間とモデル精度に関する2つの主張を検証し、注意メカニズムの実行時間とメモリのベンチマークを報告します。

-   •

訓練速度。FlashAttentionは、BERTに対するMLPerf 1.1 [Mattso20]の速度記録を15％上回り、GPT-2の速度をHuggingFace [Stateo20]の最大3$\times$およびMegatron $1.8\times$の最大[Shoeyb19]で標準的なTransformerより高速化します。FlashAttentionは長距離アリーナ（LRA）ベンチマークを2.4$\times$倍に高速化します。

-   •

品質。FlashAttentionはTransformerをより長いシーケンスにスケールし、より高い品質を実現します。FlashAttentionはコンテキスト長4KでGPT-2をMegatronのコンテキスト長1KでのGPT-2訓練よりも高速に訓練し、パープレキシティを0.7改善します。より長いシーケンスをモデル化することで、2つの長文分類タスクで6.4ポイントの向上が見られます。最後に、FlashAttentionは挑戦的なPath-Xタスク（シーケンス長16K）でランダムを上回る性能を達成できる初のTransformerを実現し、ブロックスパースFlashAttentionはPath-256（シーケンス長64K）でランダムを上回る性能を達成できることが知られている最初のシーケンスモデルを実現します。

-   •

注意機構のベンチマーク。シーケンス長に基づいてFlashAttentionとブロックスパースFlashAttentionの実行時間とメモリ性能を測定します。FlashAttentionのメモリ使用量はシーケンス長に比例してスケールし、一般的なシーケンス長（最大2K）では標準的な注意機構よりも最大で3$\times$高速であることを確認しました。ブロックスパースFlashAttentionの実行時間もシーケンス長に比例してスケールし、既存のすべての近似注意ベースラインよりも高速であることを確認しました。

追加の実験詳細は[付録E](#A5)にあります。

### 4.1 FlashAttentionによる高速モデル

##### BERT.

FlashAttentionは、我々の知る限りで最も高速なシングルノードBERTトレーニング速度を実現します。WikipediaでBERT-large [Devlin19]モデルをFlashAttentionでトレーニングしました。[表1](#table-01)は、MLPerf 1.1 [Mattso20]でトレーニング速度記録を樹立したNvidiaの実装と我々のトレーニング時間を比較しています。我々の実装は15％高速です。

<span id="table-01"></span>

![論文の表 1](../../papers/flashattention/table-01.png)

**表1.** MLPerfベンチマークが提供する同じ初期化から開始して、マスク付き言語モデリングで目標精度72.0％に達するBERT-largeの学習時間。8つの$\times$A100 GPUで10回の平均値。

##### GPT-2.

FlashAttentionは、大規模なOpenWebTextデータセット [Gokasl19]上で[Radfor19]、広く使用されているHuggingFace [Stateo20]やMegatron-LM [Shoeyb19]の実装よりもGPT-2の学習時間を短縮します。 [表2](#table-02) はHuggingfaceに対して最大3$\times$のエンドツーエンドのスピードアップ、Megatron-LMに対して1.7$\times$のスピードアップを示しています。FlashAttentionはモデル定義を変更しないため、他の2つの実装と同じパープレキシティを達成します。 [付録E](#A5) には、学習中の検証パープレキシティのプロットが含まれており、FlashAttentionがベースラインと同様に数値的に安定しており、同じ学習/検証曲線を生成することを確認しています。

<span id="table-02"></span>

![論文の表 2](../../papers/flashattention/table-02.png)

**表2.** GPT-2 スモールおよびミディアムは FlashAttention を使用することで、Huggingface の実装と比較して最大3$\times$の速度向上を達成し、Megatron-LM と比較して最大1.7$\times$の速度向上を達成します。トレーニング時間は 8$\times$A100 GPU 上で報告されています。

##### Long-range Arena.

標準 Transformer（標準実装または FlashAttention のいずれか）を Long-range Arena （LRA [Repree20]） ベンチマークで比較します。すべてのモデルの精度、スループット、およびトレーニング時間を測定します。各タスクには 1024 から 4096 の間で異なるシーケンス長があります。[Repree20] および [Xiong21] の実装と実験設定に従います。[+3] [表3](#table-03) に示すように、FlashAttention は標準アテンションと比較して最大 2.4$\times$ の速度向上を達成します。ブロックスパース FlashAttention は、テストしたすべての近似アテンション手法よりも高速です。

<span id="table-03"></span>

![論文の表 3](../../papers/flashattention/table-03.png)

**表3.** 標準アテンション、FlashAttention、ブロックスパースFlashAttention、および近似アテンションのLong-Range-Arenaベンチマークにおける性能。

### 4.2 より長いシーケンスでのより良いモデル

##### 長い文脈での言語モデリング。

FlashAttentionのランタイムとメモリ効率により、GPT-2のコンテキスト長を4に増やすことができ、$\times$ それでもMegatron-LMの最適化実装より高速に実行可能である。[表4](#table-04) は、FlashAttentionとコンテキスト長4KのGPT-2が、コンテキスト長1KのMegatron版GPT-2よりも30%高速でありながら、パープレキシティ0.7向上することを示している。

<span id="table-04"></span>

![論文の表 4](../../papers/flashattention/table-04.png)

**表4.** GPT-2 smallはFlashAttentionを使用し、Megatron-LMに比べて4$\times$長いコンテキストを持ちながら、30%速く、かつ0.7優れたパープレキシティを達成しています。8$\times$A100 GPUでのトレーニング時間が報告されています。

##### 長文文書分類

FlashAttentionを用いたより長いシーケンスでのトランスフォーマーのトレーニングは、MIMIC-III [Johnsa16]およびECtHR [Chalki19, Chalki21]データセットでの性能を向上させます。MIMIC-IIIには集中治療室の患者退院サマリーが含まれており、それぞれ複数のラベルで注釈が付けられています。ECtHRには欧州人権裁判所の法的事例が含まれており、それぞれが人権条約の違反が疑われる条文にマッピングされています。これらのデータセットの両方には非常に長いテキストドキュメントが含まれており、MIMICの平均トークン数は2,395トークンで、最長のドキュメントは14,562トークンです。一方、ECtHRの平均および最長トークン数はそれぞれ2,197および49,392です。我々は、事前学習済みのRoBERTaモデル[Liua19]のシーケンス長を増加させることによる性能向上を評価します（[Beltag20]のように位置エンベディングを繰り返します）。

[表5](#table-05)は、MIMICではシーケンス長16Kが長さ512を4.3ポイント上回り、ECtHRでは長さ8Kが長さ512を8.5ポイント上回ることを示しています。これらの差異は微妙な分布の変化によるものかもしれません：MIMIC-IIIは専門的な医療テキストを含むため、文書長の分布変化の影響を受けやすい可能性があります。一方、ECtHRは一般的な言語を含んでいます。

<span id="table-05"></span>

![論文の表 5](../../papers/flashattention/table-05.png)

**表5.** FlashAttentionを使用した場合の異なるシーケンス長での長文ドキュメント性能（マイクロ$F_{1}$）。

<span id="table-06"></span>

![論文の表 6](../../papers/flashattention/table-06.png)

**表6.** Path-XおよびPath-256で非ランダム性能を達成できる最初のトランスフォーマーモデルを報告する。

##### Path-X と Path-256。

Path-X と Path-256 ベンチマークは、長い文脈をテストするために設計された Long-Range Arena ベンチマークの難しい課題です。このタスクは、白黒 128$\times$128（または 256$\times$256）画像内の二つの点が経路でつながっているかどうかを分類するもので、画像はピクセル単位でトランスフォーマに入力されます。以前の研究では、すべてのトランスフォーマモデルはメモリ不足に陥るか、ランダムな性能しか達成できませんでした [Repree20]。長文脈をモデル化できる代替アーキテクチャの探索が行われてきました [Ref22]。ここでは、初めて Path-X と Path-256 を解決できるトランスフォーマモデルの結果を示します（[表 6](#table-06)）。トランスフォーマを Path-64 で事前学習し、その後位置埋め込みを空間的に補間することで Path-X に転送します。FlashAttention は Path-X で 61.4 の精度を達成します。さらに、ブロックスパース FlashAttention により、トランスフォーマはシーケンス長 64K までスケール可能となり、Path-256 で 63.1 の精度を達成します [+4]。

### 4.3 注意のベンチマーク

<span id="figure-03"></span>

![キャプションを参照](../../papers/flashattention/figure-03.png)

**図 3.** 左：順伝播と逆伝播のランタイム。右：注意メモリ使用量。

私たちはシーケンス長を変化させ、1つのA100 GPU（HBM 40 GB）上で、ドロップアウトとパディングマスクを使用して、FlashAttentionおよびブロックスパースFlashAttentionのランタイムとメモリ使用量をさまざまな注意ベースラインと比較測定します。正確な注意、近似注意、およびスパース注意のリファレンス実装と比較します。本文ではベースラインの一部を報告し、付録[E](#A5)には、さらに多くのベースラインと詳細が含まれています。

##### ランタイム.

[図 3](#figure-03)（左）は、FlashAttention とブロックスパース FlashAttention のフォワードおよびバックワードパスの実行時間（ミリ秒単位）を、正確、近似、スパースアテンションのベースラインと比較して報告しています（正確な数値は付録 [E]（#A5 "付録 E フル実験結果 ‣ FlashAttention： 高速でメモリ効率の良い正確なアテンション」のページ） に記載）。実行時間はシーケンス長とともに二次的に増加しますが、FlashAttention は正確なアテンションのベースラインよりも大幅に高速で、PyTorch 実装より最大で 3$\times$ 倍高速です。多くの近似/スパースアテンションメカニズムの実行時間はシーケンス長とともに線形に増加しますが、FlashAttention はメモリアクセスが少ないため、短いシーケンスでは近似およびスパースアテンションよりも依然として高速です。近似アテンションの実行時間は、シーケンス長が 512 から 1024 の間で FlashAttention と交差し始めます。一方、ブロックスパース FlashAttention は、我々が知っているすべての正確、スパース、近似アテンションの実装よりも、全シーケンス長にわたって高速です。

##### メモリ使用量。

[図3](#figure-03)（右）は、FlashAttentionおよびブロックスパースFlashAttentionのメモリ使用量を、さまざまな正確注意、近似注意、およびスパース注意のベースラインと比較したものです。FlashAttentionとブロックスパースFlashAttentionは同じメモリ使用量を持ち、シーケンス長に応じて線形に増加します。FlashAttentionは正確注意ベースラインより最大20$\times$メモリ効率が高く、近似注意ベースラインよりもメモリ効率が良いです。Linformerを除くすべてのアルゴリズムは64K前にA100 GPUでメモリ不足になりますが、FlashAttentionはLinformerよりもまだ2$\times$効率的です。

## 5 制限事項と今後の方向性

私たちは、アプローチの制限と今後の方向性について議論します。関連研究は[付録A](#A1) に記載されています。

CUDAへのコンパイル。注意のIO認識実装を構築する現在のアプローチでは、新しい注意実装ごとに新しいCUDAカーネルを書く必要があります。これは、PyTorchよりもはるかに低レベルの言語で注意アルゴリズムを書くことを要求し、かなりのエンジニアリング努力を必要とします。さらに、実装はGPUアーキテクチャ間で移植できない場合があります。これらの制限は、注意アルゴリズムを高レベル言語（例：PyTorch）で記述し、CUDAでIO認識実装にコンパイルする方法の必要性を示唆しています。これは、画像処理におけるHalideのような取り組みに似ています。[Notice13]

IO対応ディープラーニング。私たちは、IO対応のアプローチは注意機構にとどまらず拡張可能であると考えています。注意機構はトランスフォーマーにおける最もメモリ集約的な計算ですが、深層ネットワークのすべての層がGPUのHBMにアクセスします。私たちは、この研究が他のモジュールのIO対応実装の着想になることを期待しています。これらの潜在的な拡張については、[付録D](#A4)で議論しています。

マルチGPU IO対応手法。私たちのIO対応の注意実装は、単一GPUでの注意計算において定数因子内で最適解です。しかし、注意計算は複数GPUにわたって並列化可能かもしれません [Recht13]。複数GPUを使用する場合、GPU間のデータ転送を考慮したIO分析のもう一つの層が追加されます。私たちは、この研究が将来この方向での研究の刺激になることを期待しています。

#### 謝辞

私たちの実装は、ApexのFMHAコード（[https://github.com/NVIDIA/apex/tree/master/apex/contrib/csrc/fmha](https://github.com/NVIDIA/apex/tree/master/apex/contrib/csrc/fmha]）を出発点として使用しています。Young-Jun Ko氏には、FMHA実装に関する詳細な説明とCUDAに関する私たちの質問への丁寧な回答に感謝します。Sabri Eyuboglu氏、Megan Leszczynski氏、Laurel Orr氏、Yuhuai Wu氏、Beidi Chen氏、Xun Huang氏には、論文初稿に対する建設的なフィードバックと提案に感謝します。Markus Rabe氏とCharles Staats氏には、注意機構アルゴリズムに関する有益な議論に感謝します。

我々は、NIHによるU54EB020405（Mobilize）号の支援、NSFによるCCF1763315（Beyond Sparsity）、CCF1563078（Volume to Velocity）、および1937301（RTML）号の支援、ARLによるW911NF-21-2-0251（Interactive Human-AI Teaming）号の支援、ONRによるN000141712266（Unifying Weak Supervision）号およびN00014-20-1-2480（Machine Learningにおける非ユークリッド幾何学の理解と応用）、N000142012275（NEPTUNE）号の支援、NXP、Xilinx、LETI-CEA、Intel、IBM、Microsoft、NEC、Toshiba、TSMC、ARM、Hitachi、BASF、Accenture、Ericsson、Qualcomm、Analog Devices、Google Cloud、Salesforce、Total、HAI-GCP＆HAI-Azure Cloud Credits for Researchプログラム、スタンフォードデータサイエンスイニシアティブ（SDSI）、国防科学工学大学院フェローシップ（NDSEG）プログラムを通じた国防総省（DoD）およびスタンフォードDAWNプロジェクトのメンバー：Facebook、Google、VMWareの支援に深く感謝します。米国政府は、そこに記載された著作権の注記にかかわらず、政府目的で複製および配布する権限を有します。本資料で表明された意見、発見、および結論または提言は著者のものであり、必ずしもNIH、ONR、または米国政府の見解、方針、または支持（明示的または暗示的）を反映するものではありません。Atri Rudraの研究は、NSF助成金CCF-1763481により支援されています。

## 付録A 関連研究

IO対応ランタイム最適化。高速/低速メモリへの読み書きを最適化するという広い概念は、コンピュータサイエンスの歴史の中で長い間存在し、多くの名前で知られています。本稿では、I/O複雑性の分析に関する文献 [Vitter88] との最も直接的な関連を引きますが、メモリ階層の概念は基本的なものであり、ワーキングセットモデル [Dennin68] からデータ局所性 [Lama91]、演算強度のRooflineモデル [Willia09]、スケーラビリティの分析 [HotOS15]、コンピュータアーキテクチャの標準的な教科書的扱い [Approa03] に至るまで、様々な形で登場しています。本研究が、深層学習スタックのより多くの部分でこれらのアイデアを採用することをコミュニティに促すことを願っています。

構造化行列を用いた効率的な機械学習モデル。行列の乗算は、ほとんどの機械学習モデルにおける主要な計算ボトルネックです。計算量を削減するために、より効率的な行列のセットを学習するための多くのアプローチが提案されてきました。これらの行列は *構造化行列* と呼ばれ、次元 $n\times n$ に対して二次以下（$o(n^{2})$）のパラメータと実行時間を持ちます。構造化行列の最も一般的な例はスパース行列や低ランク行列、そして信号処理で一般的に見られる高速変換（フーリエ変換、チェビシェフ、多項式正弦/余弦、直交多項式など）です。機械学習において提案されている、より一般的な構造化行列のクラスには、Toeplitz型[Sindhw15]、低変位ランク[Kailat79]、準分離[Gohber99]などがあります。我々がブロックスパースアテンションに使用するバタフライパターンは、バタフライ行列[Parker95, Dao19]とその積がほぼ最適な実行時間とパラメータ数で任意の構造化行列を表現できることが示されていることに基づいています[Ref18, Dao20]。しかし、構造化行列は理論的には効率的であっても、その効率を実際の処理速度に変換することが難しいため、広く採用されていません。これは、密に制約のない行列の乗算は非常に最適化された実装があるためであり、この現象はハードウェアロト[Hooker20]として知られています。バタフライ行列[Daoa22, Daob22]の拡張は、バタフライ行列をよりハードウェアに優しいものにすることを目的としています。

スパーストレーニング。私たちのブロックスパースFlashAttentionは、スパースモデルのトレーニングをより効率的にするための一歩と見なすことができます。スパースモデルは、重み行列をスパース化することで推論（プルーニング）のためにモデルを圧縮することに成功しています[Han16, Hana15, Sanh20, Inc17, Dong17]。モデルトレーニングにおいては、ロッテリー・チケット仮説[Carbin18, Frankl19, Frankl20]は、元の密なネットワークと同じくらいの性能を発揮する、大きな密なネットワークから派生した小さなサブネットワークのセットが存在することを示唆しています。私たちのブロックスパースFlashAttentionも、アテンションの文脈では固定されたロッテリーチケットと見なすことができ、トレーニングを通じてバタフライパターンのスパース性を固定し、Long-range Arenaタスクにおいて（密な）FlashAttentionとほぼ同等の性能を発揮することを観察しています。

効率的なトランスフォーマー。トランスフォーマーベースのモデルは、自然言語処理 [Devlin19] やコンピュータビジョン [Dosovi20, Vision21] において最も広く使用されるアーキテクチャとなっています。しかし、それらの計算ボトルネックの一つは、シーケンス長に対して時間とメモリが二次的にスケーリングすることです。このボトルネックを克服するためのさまざまなアプローチがあります。ハッシュによる近似（すなわち疎性）を用いたもの（Reformer [ICML20] や Smyrf [Daras20] など）や、低ランク近似を用いたもの（Performer [Chorom20, Likhos20] など）があります。精度向上のために疎性と低ランク近似を組み合わせることもできます（例：Longformer [Beltag20]、BigBird [Systeo20]、Scatterbrain [Chen21]、Long-short transformer [Systej21]、Combiner [Ren21]）。その他のアプローチとしては、シーケンス次元に沿って圧縮し、複数のトークンに同時に注目する手法があります [ICLR19, Linguc19, ICLRb20, Refa21]。また、前のシーケンスの状態に注目して、コンテキストを延ばすことも可能です（例：Transformer-XL [Dai19] や Compressive Transformer [ICLRc20]）。詳細についてはレビュー [Taya20] を参照することをおすすめします。

モデルの長い文脈に対する注意ではなく、他のモジュールを開発するいくつかの研究があります。HiPPO [Ref20] とその拡張、特に S4 [Ref21, Ref22, Goel22] は履歴を多項式基底上で投影し、状態空間モデルを通じて履歴を正確に再構築できるようにします。これらは、CNN（効率的なトレーニング）、RNN（効率的な推論）、および連続モデル（サンプリングレートの変化に対して堅牢）の強みを組み合わせています。LambdaNetworks [Bello21]、AFT [Xivat21] および FLASH [Hua22] は、画像分類や言語モデリングの文脈で注意機構を置き換える他の試みです。

## 付録B アルゴリズムの詳細

まず、注意機構の順伝播と逆伝播を導出し、それらがメモリ効率よく計算できること（シーケンス長に対して二次ではなく線形の追加メモリを必要とする）を示します。追加メモリの量を減らすことができますが、単純には依然として二次の HBM アクセスを引き起こし、実行速度が遅くなります。順伝播と逆伝播の両方を GPU 上で実装し、HBM アクセスを削減して実行時間を短縮し、メモリフットプリントを小さくする FlashAttention アルゴリズムについて説明します。

### B.1 メモリ効率の良い順伝播

注意をメモリ効率的にする際の主な課題は、$\mathbf{K}$の列（および$\mathbf{V}$の列）を結びつけるソフトマックスです。我々のアプローチは、列を切り離すためにソフトマックスの正規化定数を別々に計算することです。この手法[Gimels18]は、文献[ICML20, Staats21]で、注意計算には二次的な*追加*メモリが必要ないことを示すために使用されてきました（ただし、HBMアクセスの回数は依然として二次的であり、実行時間は遅くなります）。

簡単のため、ここではソフトマックスの最大シフトステップを省略します。[Section B.3](#A2.SS3) の完全なアルゴリズムにはすべてのステップが含まれています。

入力シーケンス$\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}$が与えられたとき、注意出力$\mathbf{O}\in\mathbb{R}^{N\times d}$を計算したいことを思い出してください：

$$
\mathbf{S}=\mathbf{Q}\mathbf{K}^{\top}\in\mathbb{R}^{N\times N},\quad\mathbf{P}=\mathrm{softmax}(\mathbf{S})\in\mathbb{R}^{N\times N},\quad\mathbf{O}=\mathbf{P}\mathbf{V}\in\mathbb{R}^{N\times d}.
$$

我々には次のことがあります $S_{\mathrm{ij}}=q_{i}^\topk_{j}$ ここで $q_{i}$ と $k_{j}$ は、それぞれ $i$ 列と $j$ 列です $\mathbf{Q}$ と $\mathbf{K}$ の。ソフトマックスの正規化定数を定義します：

$$
L_{i}=\sum_{j}e^{q_{i}^\topk_{j}}.\tag{1}
$$

$v_{j}$を$j$番目の$\mathbf{V}$の列とすると、出力の$i$番目の列は

$$
o_{i}=P_{i:}\mathbf{V}=\sum_{j}P_{\mathrm{ij}}v_{j}=\sum_{j}\frac{e^{q_{i}^\topk_{j}}}{L_{i}}v_{j}.\tag{2}
$$

となります。$L_{i}$が計算されると、$\frac{e^{q_{i}^\topk_{j}}}{L_{i}}v_{j}$を繰り返し加算することで追加メモリなしで$o_{i}$を計算できることがわかります。したがって、順方向計算は$O(n)$の追加メモリで計算可能です：

1. 1.

すべての $i$ に対して [式 1](#A2.E1) に従って $L_{i}$ を計算します。これには $O(n)$ の追加メモリが必要です。

2. 2.

すべての $i$ に対して [式 2](#A2.E2) に従って $o_{i}$ を計算します。これには $O(d)$ の追加メモリが必要です。

### B.2 メモリ効率の良い逆伝播

注意機構の逆伝播を導出し、線形メモリで計算できることを示します。[Staats21] は、メモリ効率の良い順伝播に勾配チェックポイントを適用することで、二次的な追加メモリなしに逆伝播が可能であることを示唆しています。我々は代わりに逆伝播を明示的に導出し、どのようにメモリ効率良く計算できるかを示します。

スカラー損失関数 $\phi$ があると仮定し、出力勾配を $\mathbf{\mathrm{dO}}\in\mathbb{R}^{n\times d}$ とします（ここで $\mathbf{\mathrm{dO}}$ は $\frac{\partial\phi}{\partial\mathbf{O}}$ を表します）。入力勾配 $\mathbf{\mathrm{dQ}},\mathbf{\mathrm{dK}},\mathbf{\mathrm{dV}}\in\mathbb{R}^{n\times d}$ を計算したいと思います（ここで $\mathbf{\mathrm{dQ}},\mathbf{\mathrm{dK}},\mathbf{\mathrm{dV}}$ はそれぞれ $\frac{\partial\phi}{\partial\mathbf{Q}},\frac{\partial\phi}{\partial\mathbf{K}},\frac{\partial\phi}{\partial\mathbf{V}}$ を表します）。

勾配 $\mathbf{\mathrm{dV}}$ は簡単に理解できます。手動で逆モード自動微分（別名連鎖律）を適用すると、（行列表記で） $\mathbf{\mathrm{dV}}=\mathbf{P}^\top\mathbf{\mathrm{dO}}$ を得ます。したがって：

$$
\mathrm{dv}_{j}=\sum_{i}P_{\mathrm{ij}}\mathrm{do}_{i}=\sum_{i}\frac{e^{q_{i}^\topk_{j}}}{L_{i}}\mathrm{do}_{i}.\tag{3}
$$

すでに $L_{i}$ を計算しているので、$\mathrm{dv}_{j}$ は追加のメモリを使わずに繰り返し和を取ることで計算できます。

勾配 $\mathbf{\mathrm{dQ}}$ と $\mathbf{\mathrm{dK}}$ は少し複雑です。まず勾配 $\mathbf{\mathrm{dP}}$ と $\mathbf{\mathrm{dS}}$ を確認します。[式 2](#A2.E2) から、$\mathbf{\mathrm{dP}}=\mathbf{\mathrm{dO}}\mathbf{V}^\top$ となり、したがって：

$$
\mathrm{dP}_{\mathrm{ij}}=\mathrm{do}_{i}^\topv_{j}.
$$

$P_{i:}=\mathrm{softmax}(S_{i:})$ を思い出してください。$y=\mathrm{softmax}(x)$ のヤコビアンが $\mathrm{diag}(y)-\mathrm{yy}^\top$ であるという事実を使うと、次のようになります。

$$
\mathrm{dS}_{i:}=(\mathrm{diag}(P_{i:})-P_{i:}P_{i:}^\top)\mathrm{dP}_{i:}=P_{i:}\circ \mathrm{dP}_{i:}-(P_{i:}^\topdP_{i:})P_{i:},
$$

ここで $\circ$ は要素ごとの積を表します。

次を定義します：

$$
D_{i}=P_{i:}^\topdP_{i:}=\sum_{j}\frac{e^{q_{i}^\topk_{j}}}{L_{i}}\mathrm{do}_{i}^\topv_{j}=\mathrm{do}_{i}^\top\sum_{j}\frac{e^{q_{i}^{\top}k_{j}}}{L_{i}}v_{j}=\mathrm{do}_{i}^\topo_{i},\tag{4}
$$

すると

$$
\mathrm{dS}_{i:}=P_{i:}\circ \mathrm{dP}_{i:}-D_{i}P_{i:}.
$$

ゆえに

$$
\mathrm{dS}_{\mathrm{ij}}=P_{\mathrm{ij}}\mathrm{dP}_{\mathrm{ij}}-D_{i}P_{\mathrm{ij}}=P_{\mathrm{ij}}(\mathrm{dP}_{\mathrm{ij}}-D_{i}).
$$

これで勾配 $\mathbf{\mathrm{dQ}}$ と $\mathbf{\mathrm{dK}}$ を取得できます。$S_{\mathrm{ij}}=q_{i}^\topk_{j}$ を思い出してください。次に、

$$
\mathrm{dq}_{i}=\sum_{j}\mathrm{dS}_{\mathrm{ij}}k_{j}=\sum_{j}P_{\mathrm{ij}}(\mathrm{dP}_{\mathrm{ij}}-D_{i})k_{j}=\sum_{j}\frac{e^{q_{i}^\topk_{j}}}{L_{i}}(\mathrm{do}_{i}^\topv_{j}-D_{i})k_{j}.\tag{5}
$$

同様に、

$$
\mathrm{dk}_{j}=\sum_{i}\mathrm{dS}_{\mathrm{ij}}q_{i}=\sum_{i}P_{\mathrm{ij}}(\mathrm{dP}_{\mathrm{ij}}-D_{i})q_{i}=\sum_{i}\frac{e^{q_{i}^\topk_{j}}}{L_{i}}(\mathrm{do}_{i}^\topv_{j}-D_{i})q_{i}.\tag{6}
$$

したがって逆伝播は追加メモリ $O(n)$ を使っても計算できます：

1. 1.

すべての $j$ に対して [Eq. 3](#A2.E3) に従って $\mathrm{dv}_{j}$ を計算します。これには $O(d)$ の追加メモリが必要です。

2. 2.

すべての $i$ に対して [Eq. 4](#A2.E4) に従って $D_{i}$ を計算します。これには $O(n)$ の追加メモリが必要です。

3.  3.

すべての $i$ に対して [Eq. 5](#A2.E5) に従って $\mathrm{dq}_{i}$ を計算します。これには $O(d)$ の追加メモリが必要です。

4.  4.

すべての $j$ に対して [Eq. 6](#A2.E6) に従って $\mathrm{dk}_{j}$ を計算します。これには $O(d)$ の追加メモリが必要です。

### B.3 FlashAttention： 順方向伝播

FlashAttention のフォワードパスの詳細を説明します。入力シーケンス $\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}$ が与えられた場合、私たちは注意出力 $\mathbf{O}\in\mathbb{R}^{N\times d}$ を計算したいです：

$$
\mathbf{S}=\tau\mathbf{Q}\mathbf{K}^{\top}\in\mathbb{R}^{N\times N},\quad\mathbf{S}^{\mathrm{masked}}=\mathrm{mask}(S)\in\mathbb{R}^{N\times N},\quad\mathbf{P}=\mathrm{softmax}(\mathbf{S}^{\mathrm{masked}})\in\mathbb{R}^{N\times N},
$$

$$
\mathbf{P}^{\mathrm{dropped}}=\mathrm{dropout}(\mathbf{P},p_{\mathrm{drop}}),\quad\mathbf{O}=\mathbf{P}^{\mathrm{dropped}}\mathbf{V}\in\mathbb{R}^{N\times d},
$$

ここで $\tau\in\mathbb{R}$ はいくつかのソフトマックススケーリング（通常 $\frac{1}{\sqrt{d}}$）、mask は入力のいくつかの要素を $-\infty$ に設定し、その他の要素は同じままにするマスキング関数（例えば、バッチ内のシーケンスの長さが同じでなくパディングされている場合のキー・パディング・マスク）、そして $\mathrm{dropout}(x,p)$ は $x$ に要素ごとにドロップアウトを適用する（つまり、各要素 $x$ ごとに確率 $1-p$ で $\frac{x}{1-p}$ を出力し、確率 $p$ で 0 を出力する）。

完全なアルゴリズムは [Algorithm 2](#alg2) にあります。出力 $\mathbf{O}$、ソフトマックス統計 $\ell$ および $m$、そして擬似乱数生成器の状態 ${\cal R}$ を逆伝播用に保存します。

<span id="alg2"></span>

**アルゴリズム 2: FlashAttention フォワードパス**

- **入力:** HBM 内の行列 $\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}$、サイズ $M$ のオンチップ SRAM、ソフトマックススケーリング定数 $\tau\in\mathbb{R}$、マスキング関数 mask、ドロップアウト確率 $p_{\mathrm{drop}}$。
- 疑似乱数生成器の状態${\cal R}$を初期化し、HBMに保存。
- ブロックサイズを $B_{c}=\left\lceil\frac{M}{4d}\right\rceil,B_{r}=\min\left(\left\lceil\frac{M}{4d}\right\rceil,d\right)$ に設定する。
- HBM 内に $\mathbf{O}=(0)_{N\times d}\in\mathbb{R}^{N\times d},\ell=(0)_{N}\in\mathbb{R}^{N},m=(-\infty)_{N}\in\mathbb{R}^{N}$ を初期化。
- $\mathbf{Q}$ を $T_{r}=\left\lceil\frac{N}{B_{r}}\right\rceil$ 個のブロック $\mathbf{Q}_{1},\dots,\mathbf{Q}_{T_{r}}$ に分け、それぞれのサイズを $B_{r}\times d$ にして、 $\mathbf{K},\mathbf{V}$ を $T_{c}=\left\lceil\frac{N}{B_{c}}\right\rceil$ 個のブロック $\mathbf{K}_{1},\dots,\mathbf{K}_{T_{c}}$ と $\mathbf{V}_{1},\dots,\mathbf{V}_{T_{c}}$ に分け、それぞれのサイズを $B_{c}\times d$ にする。
- $\mathbf{O}$ を $T_{r}$ 個のブロック $\mathbf{O}_{i},\dots,\mathbf{O}_{T_{r}}$ に分け、それぞれのサイズを $B_{r}\times d$ にする。$\ell$ を $T_{r}$ 個のブロック $\ell_{i},\dots,\ell_{T_{r}}$ に分け、それぞれのサイズを $B_{r}$ にする。$m$ を $T_{r}$ 個のブロック $m_{1},\dots,m_{T_{r}}$ に分け、それぞれのサイズを $B_{r}$ にする。
- **$1\leq j\leq T_{c}$について実行:**
  - HBMからオンチップSRAMに$\mathbf{K}_{j},\mathbf{V}_{j}$をロードする。
  - **$1\leq i\leq T_{r}$ に対して行う:**
    - $\mathbf{Q}_{i},\mathbf{O}_{i},\ell_{i},m_{i}$ を HBM からオンチップ SRAM にロードする。
    - チップ上で $\mathbf{S}_{\mathrm{ij}}=\tau\mathbf{Q}_{i}\mathbf{K}_{j}^\top\in\mathbb{R}^{B_{r}\times B_{c}}$ を計算します。
    - チップ上で $\mathbf{S}_{\mathrm{ij}}^{\mathrm{masked}}=\mathrm{mask}(\mathbf{S}_{\mathrm{ij}})$ を計算します。
    - チップ上で、$\tilde{m}_{\mathrm{ij}}=\mathrm{rowmax}(\mathbf{S}_{\mathrm{ij}}^{\mathrm{masked}})\in\mathbb{R}^{B_{r}}$、$\tilde{\mathbf{P}}_{\mathrm{ij}}=\exp(\mathbf{S}_{\mathrm{ij}}^{\mathrm{masked}}-\tilde{m}_{\mathrm{ij}})\in\mathbb{R}^{B_{r}\times B_{c}}$（ポイントごと）、$\tilde{\ell}_{\mathrm{ij}}=\mathrm{rowsum}(\tilde{\mathbf{P}}_{\mathrm{ij}})\in\mathbb{R}^{B_{r}}$ を計算する。
    - チップ上で、$m_{i}^{\mathrm{new}}=\max(m_{i},\tilde{m}_{\mathrm{ij}})\in\mathbb{R}^{B_{r}}$、$\ell_{i}^{\mathrm{new}}=e^{m_{i}-m_{i}^{\mathrm{new}}}\ell_{i}+e^{\tilde{m}_{\mathrm{ij}}-m_{i}^{\mathrm{new}}}\tilde{\ell}_{\mathrm{ij}}\in\mathbb{R}^{B_{r}}$ を計算する。
    - チップ上で、$\tilde{\mathbf{P}}_{\mathrm{ij}}^{\mathrm{dropped}}=\mathrm{dropout}(\tilde{\mathbf{P}}_{\mathrm{ij}},p_{\mathrm{drop}})$ を計算する。
    - $\mathbf{O}_{i}\leftarrow\mathrm{diag}(\ell_{i}^{\mathrm{new}})^{-1}(\mathrm{diag}(\ell_{i})e^{m_{i}-m_{i}^{\mathrm{new}}}\mathbf{O}_{i}+e^{\tilde{m}_{\mathrm{ij}}-m_{i}^{\mathrm{new}}}\tilde{\mathbf{P}}_{\mathrm{ij}}^{\mathrm{dropped}}\mathbf{V}_{j})$ を HBM に書き込む。
    - $\ell_{i}\leftarrow\ell_{i}^{\mathrm{new}}$、$m_{i}\leftarrow m_{i}^{\mathrm{new}}$ を HBM に書き込む。
- **返却:** $\mathbf{O},\ell,m,{\cal R}$ を返す。

### B.4 FlashAttention： Backward パス

FlashAttention の逆伝播の詳細を説明する。入力シーケンス $\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}$、出力 $\mathbf{O}\in\mathbb{R}^{N\times d}$、および出力勾配 $\mathbf{\mathrm{dO}}$ が与えられた場合、入力勾配 $\mathbf{\mathrm{dQ}},\mathbf{\mathrm{dK}},\mathbf{\mathrm{dV}}\in\mathbb{R}^{N\times d}$ を計算したい。

完全性のために、まず[アルゴリズム3](#alg3)で標準的なアテンションの逆伝播を説明します。

<span id="alg3"></span>

**アルゴリズム 3: 標準アテンション逆伝播**

- **入力:** 行列$\mathbf{Q},\mathbf{K},\mathbf{V},\mathbf{\mathrm{dO}}\in\mathbb{R}^{N\times d}$、HBMにおける$\mathbf{P}\in\mathbb{R}^{N\times N}$。
- HBMからブロック単位で$\mathbf{P},\mathbf{\mathrm{dO}}$をロードし、$\mathbf{\mathrm{dV}}=\mathbf{P}^{\top}\mathbf{\mathrm{dO}}\in\mathbb{R}^{N\times d}$を計算し、HBMに$\mathbf{\mathrm{dV}}$を書き込みます。
- HBMからブロック単位で$\mathbf{\mathrm{dO}},\mathbf{V}$を読み込み、$\mathbf{\mathrm{dP}}=\mathbf{\mathrm{dO}}\mathbf{V}^{\top}\in\mathbb{R}^{N\times N}$を計算し、HBMに$\mathbf{\mathrm{dP}}$を書き込む。
- HBMから$\mathbf{P},\mathbf{\mathrm{dP}}$を読み込み、$\mathrm{dS}_{\mathrm{ij}}=P_{\mathrm{ij}}(\mathrm{dP}_{\mathrm{ij}}-\sum_{l}P_{\mathrm{il}}\mathrm{dP}_{\mathrm{il}})$ $\mathbf{\mathrm{dS}}\in\mathbb{R}^{N\times N}$を計算し、HBMに$\mathbf{\mathrm{dS}}$を書き込む。
- HBMからブロック単位で$\mathbf{\mathrm{dS}}$・$\mathbf{K}$を読み込み、計算$\mathbf{\mathrm{dQ}}=\mathbf{\mathrm{dS}}\mathbf{K}$、HBMに$\mathbf{\mathrm{dQ}}$を書き込む。
- HBMからブロック単位で$\mathbf{\mathrm{dS}}$・$\mathbf{Q}$を読み込み、$\mathbf{\mathrm{dK}}=\mathbf{\mathrm{dS}}^{\top}\mathbf{Q}$を計算し、HBMに$\mathbf{\mathrm{dK}}$を書き込む。
- **返却:** $\mathbf{\mathrm{dQ}},\mathbf{\mathrm{dK}},\mathbf{\mathrm{dV}}$を返す。

次に FlashAttention の逆伝播について 2 つの観察を行います：

1. 1.

フォワードパスからのサイズ $O(N^{2})$ のドロップアウトマスクを保存する必要はありません。代わりに、フォワードパスの疑似乱数生成器の状態を保存し、逆伝播でドロップアウトマスクを再生成できます。これにより、$O(N)$ の追加メモリのみを使用できます。

2. 2.

ソフトマックスの勾配を計算する際、[式4](#A2.E4) を使用して、サイズが$N$の$P_{i:}$および$\mathrm{dP}_{i:}$にわたって集約せずに$D_{i}=P_{i:}^{\top}\mathrm{dP}_{i:}$を計算します（これらはSRAMに収まらない可能性があります）。代わりに$D_{i}=\mathrm{do}_{i}^{\top}o_{i}$を書き換え、サイズ$d$のベクトル間でドット積を計算することができます。

フル FlashAttention の逆伝播アルゴリズムは [Algorithm 4](#alg4) にあります。概念的には、これは [Section B.2](#A2.SS2) の導出のブロック版に過ぎません。

<span id="alg4"></span>

**アルゴリズム 4: FlashAttention 逆伝播**

- **入力:** HBM上の行列$\mathbf{Q},\mathbf{K},\mathbf{V},\mathbf{O},\mathbf{\mathrm{dO}}\in\mathbb{R}^{N\times d}$、HBM上のベクトル$\ell,m\in\mathbb{R}^{N}$、サイズ$M$のオンチップSRAM、ソフトマックススケーリング定数$\tau\in\mathbb{R}$、マスキング関数mask、ドロップアウト確率$p_{\mathrm{drop}}$、順伝播からの擬似乱数生成器の状態${\cal R}$。
- 疑似乱数生成器の状態を ${\cal R}$ に設定する。
- ブロックサイズを $B_{c}=\left\lceil\frac{M}{4d}\right\rceil,B_{r}=\min\left(\left\lceil\frac{M}{4d}\right\rceil,d\right)$ に設定する。
- $\mathbf{Q}$ を $T_{r}=\left\lceil\frac{N}{B_{r}}\right\rceil$ ブロック $\mathbf{Q}_{1},\dots,\mathbf{Q}_{T_{r}}$ に分割し、それぞれのサイズは $B_{r}\times d$ とし、$\mathbf{K},\mathbf{V}$ を $T_{c}=\left\lceil\frac{N}{B_{c}}\right\rceil$ ブロック $\mathbf{K}_{1},\dots,\mathbf{K}_{T_{c}}$ および $\mathbf{V}_{1},\dots,\mathbf{V}_{T_{c}}$ に分割し、それぞれのサイズは $B_{c}\times d$ とする。
- $\mathbf{O}$をサイズ$B_{r}\times d$の$\mathbf{O}_{i},\dots,\mathbf{O}_{T_{r}}$ブロックの$T_{r}$に分割し、$\mathbf{\mathrm{dO}}$をサイズ$B_{r}\times d$の$\mathbf{\mathrm{dO}}_{i},\dots,\mathbf{\mathrm{dO}}_{T_{r}}$ブロックの$T_{r}$に分割し、$\ell$をサイズ$B_{r}$の$\ell_{i},\dots,\ell_{T_{r}}$ブロックの$T_{r}$に分割し、$m$をサイズ$B_{r}$の$m_{1},\dots,m_{T_{r}}$ブロックの$T_{r}$に分割する。
- HBMで$\mathbf{\mathrm{dQ}}=(0)_{N\times d}$を初期化し、それを$T_{r}$ブロック$\mathbf{\mathrm{dQ}}_{1},\dots,\mathbf{\mathrm{dQ}}_{T_{r}}$に、各$B_{r}\times d$のサイズに分割する。HBMで$\mathbf{\mathrm{dK}}=(0)_{N\times d},\mathbf{\mathrm{dV}}=(0)_{N\times d}$を初期化し、$\mathbf{\mathrm{dK}},\mathbf{\mathrm{dV}}$を$T_{c}$ブロック$\mathbf{\mathrm{dK}}_{1},\dots,\mathbf{\mathrm{dK}}_{T_{c}}$および$\mathbf{\mathrm{dV}}_{1},\dots,\mathbf{\mathrm{dV}}_{T_{c}}$に、各$B_{c}\times d$のサイズに分割する。
- **$1\leq j\leq T_{c}$について実行:**
  - HBMからオンチップSRAMに$\mathbf{K}_{j},\mathbf{V}_{j}$をロードする。
  - SRAM上で$\tilde{\mathbf{\mathrm{dK}}}_{j}=(0)_{B_{c}\times d},\tilde{\mathbf{\mathrm{dV}}}_{j}=(0)_{B_{c}\times d}$を初期化する。
  - **$1\leq i\leq T_{r}$について実行:**
    - HBMからオンチップSRAMに$\mathbf{Q}_{i},\mathbf{O}_{i},\mathbf{\mathrm{dO}}_{i},\mathbf{\mathrm{dQ}}_{i},\ell_{i},m_{i}$をロードする。
    - オンチップで$\mathbf{S}_{\mathrm{ij}}=\tau\mathbf{Q}_{i}\mathbf{K}_{j}^\top\in\mathbb{R}^{B_{r}\times B_{c}}$を計算する。
    - オンチップで$\mathbf{S}_{\mathrm{ij}}^{\mathrm{masked}}=\mathrm{mask}(\mathbf{S}_{\mathrm{ij}})$を計算する。
    - オンチップで$\mathbf{P}_{\mathrm{ij}}=\mathrm{diag}(l_{i})^{-1}\exp(\mathbf{S}_{\mathrm{ij}}^{\mathrm{masked}}-m_{i})\in\mathbb{R}^{B_{r}\times B_{c}}$を計算する。
    - チップ上で、各要素が確率$1-p_{\mathrm{drop}}$で値$\frac{1}{1-p_{\mathrm{drop}}}$を持ち、確率$p_{\mathrm{drop}}$で値0を持つドロップアウトマスク$\mathbf{Z}_{\mathrm{ij}}\in\mathbb{R}^{B_{r}\times B_{c}}$を計算します。
    - チップ上で、$\mathbf{P}_{\mathrm{ij}}^{\mathrm{dropped}}=\mathbf{P}_{\mathrm{ij}}\circ\mathbf{Z}_{\mathrm{ij}}$（要素ごとの掛け算）を計算します。
    - チップ上で、$\tilde{\mathbf{\mathrm{dV}}_{j}}\leftarrow\tilde{\mathbf{\mathrm{dV}}_{j}}+(\mathbf{P}_{\mathrm{ij}}^{\mathrm{dropped}})^{\top}\mathbf{\mathrm{dO}}_{i}\in\mathbb{R}^{B_{c}\times d}$を計算します。
    - チップ上で、$\mathbf{\mathrm{dP}}_{\mathrm{ij}}^{\mathrm{dropped}}=\mathbf{\mathrm{dO}}_{i}\mathbf{V}_{j}^{\top}\in\mathbb{R}^{B_{r}\times B_{c}}$を計算します。
    - チップ上で、$\mathbf{\mathrm{dP}}_{\mathrm{ij}}=\mathbf{\mathrm{dP}}_{\mathrm{ij}}^{\mathrm{dropped}}\circ\mathbf{Z}_{\mathrm{ij}}$（要素ごとの掛け算）を計算します。
    - チップ上で、$D_{i}=\mathrm{rowsum}(\mathbf{\mathrm{dO}}_{i}\circ\mathbf{O}_{i})\in\mathbb{R}^{B_{r}}$を計算します。
    - チップ上で、$\mathbf{\mathrm{dS}}_{\mathrm{ij}}=\mathbf{P}_{\mathrm{ij}}\circ(\mathbf{\mathrm{dP}}_{\mathrm{ij}}-D_{i})\in\mathbb{R}^{B_{r}\times B_{c}}$を計算します。
    - $\mathbf{\mathrm{dQ}}_{i}\leftarrow\mathbf{\mathrm{dQ}}_{i}+\tau\mathbf{\mathrm{dS}}_{\mathrm{ij}}\mathbf{K}_{j}\in\mathbb{R}^{B_{r}\times d}$をHBMに書き込みます。
    - チップ上で $\tilde{\mathbf{\mathrm{dK}}}_{j}\leftarrow\tilde{\mathbf{\mathrm{dK}}}_{j}+\tau\mathbf{\mathrm{dS}}_{\mathrm{ij}}^{\top}\mathbf{Q}_{i}\in\mathbb{R}^{B_{c}\times d}$ を計算する。
  - $\mathbf{\mathrm{dK}}_{j}\leftarrow\tilde{\mathbf{\mathrm{dK}}_{j}},\mathbf{\mathrm{dV}}_{j}\leftarrow\tilde{\mathbf{\mathrm{dV}}_{j}}$ を HBM に書き込む。
- **返却:** $\mathbf{\mathrm{dQ}},\mathbf{\mathrm{dK}},\mathbf{\mathrm{dV}}$ を返す。

前向きパスと同様に、後向きパスは $O(N^{2})$ FLOPs を実行し、入力、出力、出力勾配、入力勾配に加えて $O(N)$ の追加メモリのみを必要とすることがわかる。

前向きパスと同様に、後向きパスの IO 複雑性を解析する （[定理 2](#Thmtheorem2)）。

###### 定理 5.

$N$ をシーケンス長、$d$ をヘッド次元、$M$ を $d\leq M\leq \mathrm{Nd}$ での SRAM サイズとする。標準的なアテンション（[アルゴリズム](#alg0)）の逆伝播では $\Theta(\mathrm{Nd}+N^{2})$ 回の HBM アクセスが必要であるのに対し、FlashAttention の逆伝播（[アルゴリズム 4](#alg4)）では $\Theta(N^{2}d^{2}M^{-1})$ 回の HBM アクセスが必要である。

証明は[付録 C](#A3)にある。

### B.5 [Staats21] との比較

ここでは、我々の FlashAttention アルゴリズムと [Staats21] のアルゴリズムの類似点と相違点について述べる。

概念的には、FlashAttention と [Staats21] の両方は、定着したタイル化（またはソフトマックススケーリング）手法 [Gimels18, ICML20] を用いてアテンション行列のブロックで動作する。メモリ使用量を減らすために、両方の方法とも順伝播で大きなアテンション行列を保存せず、逆伝播で再計算する。

最初の大きな違いは、[Staats21] が総メモリ使用量（必要なGPUメモリの最大量）の削減に焦点を当てているのに対し、FlashAttention はメモリアクセス（メモリの読み書き回数）の削減に焦点を当てている点です。[セクション 2](#S2) で述べられているように、メモリアクセス量は実行時間を決定する主な要因です。メモリアクセスを減らすことは、必然的に必要な総メモリ量の削減にもつながります（例えば、ある操作が $A$ 回のメモリアクセスを行う場合、その総メモリ要件は最大で $A$ です）。その結果、FlashAttention は標準的なアテンションよりも高速（2-4$\times$）である一方、[Staats21] は標準的なアテンションとほぼ同じ速度か、やや遅いです。必要な総メモリ量の観点では、両方の手法とも大幅なメモリ節約を提供します。

2つの方法の2つ目の違いは、各ブロックから次のブロックに渡す情報の要約方法です。[Staats21]は、一時的な出力とソフトマックス正規化統計量を用いて各ブロックを要約します。順方向伝播の最後に、すべてのブロックの一時的な出力が統計量を用いて組み合わされ、最終出力が生成されます。代わりにFlashAttentionは、各ブロックを処理した後に出力を逐次更新するため（[アルゴリズム 1](#alg1) 行 [12](#alg1.l12)）、出力のコピーは1つだけで済みます（$K$ブロックの場合の$K$コピーの代わりに）。これは、FlashAttentionが[Staats21]と比べて総メモリ要件が小さいことを意味します。

最後の大きな違いは、バックワードパスの計算方法です。[Staats21] は勾配チェックポイントを使用して、各ブロックの注意行列と一時的な出力を再計算します。一方で FlashAttention は、バックワードパスを解析的に簡略化します（[セクション B.2](#A2.SS2) および [B.4](#A2.SS4)）。FlashAttention は注意行列のみを再計算し、各ブロックの一時的な出力は再計算しません。これによりバックワードパスのメモリ要件が減少し、速度向上が得られます。

## 付録 C 証明

###### [定理 1](#Thmtheorem1) の証明

まず、必要な FLOPs の数と追加メモリを数えます。

支配的な FLOPs は行列乗算から発生します。内部ループでは、（[アルゴリズム 1](#alg1) の行 [9](#alg1.l9)）、$\mathbf{Q}_{i}\in\mathbb{R}^{B_{r}\times d}$ および $\mathbf{K}_{j}\in\mathbb{R}^{B_{c}\times d}$ に対して $\mathbf{Q}_{i}\mathbf{K}_{j}^{\top}\in\mathbb{R}^{B_{r}\times B_{c}}$ を計算し、これには $O(B_{r}B_{c}d)$ FLOPs がかかります。また、（[アルゴリズム 1](#alg1) の行 [12](#alg1.l12)）、$\tilde{\mathbf{P}}_{\mathrm{ij}}\in\mathbb{R}^{B_{r}\times B_{c}}$ および $\mathbf{V}_{j}\in\mathbb{R}^{B_{c}\times d}$ に対して $\tilde{\mathbf{P}}_{\mathrm{ij}}\mathbf{V}_{j}\in\mathbb{R}^{B_{r}\times d}$ を計算し、これには $O(B_{r}B_{c}d)$ FLOPs がかかります。内部ループは $T_{c}T_{r}=\left\lceil\frac{N}{B_{c}}\right\rceil\left\lceil\frac{N}{B_{r}}\right\rceil$ 回実行されます。したがって、FLOPs の総数は

$$
O\left(\frac{N^{2}}{B_{c}B_{r}}B_{r}B_{c}d\right)=O(N^{2}d).
$$

追加で必要なメモリに関しては、統計情報 $(\ell,m)$ を格納するために $O(N)$ のメモリが必要であることがわかります。

私たちは今、$0\leq j\leq T_{c}$ に対する $j$ に関して帰納法によってアルゴリズムの正しさを証明します。$\mathbf{K}$ の最初の $\mathrm{jB}_{c}$ 行を $\mathbf{K}_{:j}\in\mathbb{R}^{\mathrm{jB}_{c}\times d}$ とし、同様に $\mathbf{V}$ の最初の $\mathrm{jB}_{c}$ 行を $\mathbf{V}_{:j}\in\mathbb{R}^{\mathrm{jB}_{c}\times d}$ とします。$\mathbf{S}_{:,:j}=\mathbf{Q}\mathbf{K}_{:j}^{\top}\in\mathbb{R}^{N\times \mathrm{jB}_{c}}$ および $\mathbf{P}_{:,:j}=\mathrm{softmax}(\mathbf{S}_{:,:j})\in\mathbb{R}^{N\times \mathrm{jB}_{c}}$（行ごとにソフトマックスを適用）とします。$j$ 回目の外側ループの後の HBM における $m,\ell,\mathbf{O}$ の値を $m^{j},\ell^{(j)},\mathbf{O}^{(j)}$ とします（[アルゴリズム 1](#alg1) の行 [5](#alg1.l5)）。（これらの $m,\ell,\mathbf{O}$ の値は、外側ループの各イテレーション後に更新されることに注意してください。）外側ループの $j$ 回目のイテレーション後に、HBM 内で次の計算が行われていることを示したい：

$$
m^{(j)}=\mathrm{rowmax}(\mathbf{S}_{:,:j})\in\mathbb{R}^{N},\quad\ell^{(j)}=\mathrm{rowsum}(\exp(\mathbf{S}_{:,:j}-m^{(j)}))\in\mathbb{R}^{N},\quad\mathbf{O}^{(j)}=\mathbf{P}_{:,:j}\mathbf{V}_{:j}\in\mathbb{R}^{N\times d}.
$$

初期化（[アルゴリズム1]（#alg1「3.1における効率的な注意アルゴリズムとタイルと再計算 ‣ 3 FlashAttention： アルゴリズム、解析、拡張 ‣ FlashAttention： 高速かつメモリ効率の正確な注意とIO認識」）の行[2]（#alg1.l2 「アルゴリズム1 ‣ 3.1 タイルと再計算を伴う効率的な注意アルゴリズム ‣ FlashAttention：アルゴリズム、解析、拡張 ‣ FlashAttention：高速かつメモリ効率の正確な注意IO-Awareness）を活用し、この主張は$j=0$（すなわち外側ループのいかなる反復も実行される前）に当てはまります。ある$j=0,\dots,T_{c}-1$、その主張が成り立つと仮定します。この主張が$j+1$にも成り立つことを示したいのです。実際、内ループの統計を更新すると（[アルゴリズム1]（#alg1「3.1におけるタイルと再計算を伴う効率的な注意アルゴリズム ‣ 3 FlashAttention： アルゴリズム、解析、拡張 ‣ FlashAttention： 高速かつメモリ効率の正確な注意とIO認識」）の行[10]（#alg1.l10 「アルゴリズム1 ‣ 3.1 タイルと再計算を伴う効率的な注意アルゴリズム ‣ 3 FlashAttention：アルゴリズム、解析」 拡張 ‣ FlashAttention： Fast and Memory-Efficient Exact Attention with IO-Awareness））は、外側ループの$(j+1)$回目の反復で、$\tilde{m}\in\mathbb{R}^{N}$ が$\mathbf{S}_{:,j:j+1}$列の行最大値（列$\mathrm{jB}_{c}$から列$(j+1)B_{c}-1$までの$\mathbf{S}$スライス）である$m^{(j+1)}=\max(m^{(j)},\tilde{m})$を更新します。これは次のことを意味します。

$$
m^{(j+1)}=\mathrm{rowmax}(\mathbf{S}_{:,:j+1})\in\mathbb{R}^{N}.
$$

同様に、我々は更新します

$$
\ell^{(j+1)}=e^{m^{(j)}-m^{(j+1)}}\ell^{(j)}+e^{\tilde{m}-m^{(j+1)}}\tilde{\ell},
$$

は $\tilde{\ell}=\mathrm{rowsum}(\exp(\mathbf{S}_{:,j:j+1}-\tilde{m}))\in\mathbb{R}^{N}$ である。[セクション 3.1](#S3.SS1) の同じ代数的操作によって、次を得る：

$$
\ell^{(j+1)}=\mathrm{rowsum}(\exp(\mathbf{S}_{:,:j+1}-m^{(j+1)}))\in\mathbb{R}^{N}.
$$

$\mathbf{V}_{j:j+1}$ を $\mathbf{V}$ の $\mathrm{jB}_{c}$ 列から $(j+1)B_{c}-1$ 列までのスライスとし、以下も更新します：

$$
\mathbf{O}^{(j+1)}\qquad =\mathrm{diag}(\ell^{(j+1)})^{-1}(\mathrm{diag}(\ell^{(j)})e^{m^{(j)}-m^{(j+1)}}\mathbf{O}^{(j)}+e^{\tilde{m}-m^{(j+1)}}\exp(\mathbf{S}_{j:j+1}-\tilde{m})\mathbf{V}_{j:j+1})
$$

$$
=\mathrm{diag}(\ell^{(j+1)})^{-1}(\mathrm{diag}(\ell^{(j)})e^{m^{(j)}-m^{(j+1)}}\mathbf{P}_{:,:j}\mathbf{V}_{:j}+e^{-m^{(j+1)}}\exp(\mathbf{S}_{j:j+1})\mathbf{V}_{j:j+1})
$$

$$
=\mathrm{diag}(\ell^{(j+1)})^{-1}(\mathrm{diag}(\ell^{(j)})e^{m^{(j)}-m^{(j+1)}}\mathrm{diag}(\ell^{(j)})\exp(\mathbf{S}_{:,:j}-m^{(j)})\mathbf{V}_{:j}+e^{-m^{(j+1)}}\exp(\mathbf{S}_{j:j+1})\mathbf{V}_{j:j+1})
$$

$$
=\mathrm{diag}(\ell^{(j+1)})^{-1}(e^{-m^{(j+1)}}\exp(\mathbf{S}_{:,:j})\mathbf{V}_{:j}+e^{-m^{(j+1)}}\exp(\mathbf{S}_{j:j+1})\mathbf{V}_{j:j+1})
$$

$$
=\mathrm{diag}(\ell^{(j+1)})^{-1}(\exp(\mathbf{S}_{:,:j}-m^{(j+1)})\mathbf{V}_{:j}+\exp(\mathbf{S}_{j:j+1}-m^{(j+1)})\mathbf{V}_{j:j+1})
$$

$$
=\mathrm{diag}(\ell^{(j+1)})^{-1}\left(\exp\left(\begin{bmatrix}\mathbf{S}_{:,:j}&\mathbf{S}_{j:j+1}\end{bmatrix}-m^{(j+1)}\right)\right)\begin{bmatrix}\mathbf{V}_{:j}\\
\mathbf{V}_{j:j+1}\end{bmatrix}
$$

$$
=\mathrm{softmax}(\mathbf{S}_{:j+1})\mathbf{V}_{:j+1}.
$$

このとき、主張は $j+1$ に対しても成り立つことがわかる。帰納法により、すべての $j=0,\dots,T_{c}$ に対して主張は成り立つ。

$j=T_{c}$ のとき、HBM 内の $\mathbf{O}$ の最終値は $\mathrm{softmax}(\mathbf{S})\mathbf{V}=\mathrm{softmax}(\mathbf{Q}\mathbf{K}^{\top})\mathbf{V}$ であると結論される。

∎

###### [定理 2](#Thmtheorem2) の証明。

まず、標準の注意実装の IO 複雑度を解析する。入力 $\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}$ は HBM にあり、アルゴリズムの終了時に出力 $\mathbf{O}\in\mathbb{R}^{N\times d}$ が HBM に書き込まれる。

行列乗算 $\mathbf{S}=\mathbf{Q}\mathbf{K}^{\top}$ を計算する最初のステップでは、入力 $\mathbf{Q},\mathbf{K}$ が HBM から読み取られ、出力 $\mathbf{S}\in\mathbb{R}^{N\times N}$ が HBM に書き込まれます（[アルゴリズム](#alg0) の [1行目](#alg0.l1)）。これにより $\Theta(\mathrm{Nd}+N^{2})$ 回の HBM アクセスが発生します。

行列 $\mathbf{P}=\mathrm{softmax}(\mathbf{S})$ を計算する2番目のステップでは、入力 $\mathbf{S}$ が HBM から読み取られ、出力 $\mathbf{P}$ が HBM に書き込まれます（[アルゴリズム](#alg0) 行 [2](#alg0.l2)）。これにより $\Theta(N^{2})$ 回の HBM アクセスが発生します。

$\mathbf{O}=\mathbf{P}\mathbf{V}$ を計算する最後のステップでは、入力 $\mathbf{P},\mathbf{V}$ がグローバルメモリから読み取られ、出力 $\mathbf{O}$ が HBM に書き込まれます（[アルゴリズム](#alg0) の [3行目](#alg0.l3)）。これにより $\Theta(\mathrm{Nd}+N^{2})$ 回の HBM アクセスが発生します。

全体として、標準的なアテンション実装では $\Theta(\mathrm{Nd}+N^{2})$ 回のグローバルメモリアクセスが必要です。

我々は次にストリーミングアテンションの IO 複雑性を分析します。

は [アルゴリズム 1](#alg1) に従って、$\mathbf{K}$ および $\mathbf{V}$ の各要素が一度 HBM からロードされることがわかる （[アルゴリズム 1](#alg1) の [6行目](#alg1.l6)）。私たちは $\mathbf{Q}$ および $\mathbf{O}$ に対して $T_{c}$ 回のパスを行い、各パスで $\mathbf{Q}$ および $\mathbf{O}$ のすべてを HBM にロードする （[アルゴリズム 1](#alg1) の [8行目](#alg1.l8)）。したがって、HBM アクセスの回数は $\Theta\left(\mathrm{Nd}+\mathrm{NdT}_{c}\right)=\Theta(\mathrm{NdT}_{c})$ である。

ブロックサイズ $B_{c}$ および $B_{r}$ に関する条件を導出します。ブロック $\mathbf{K}_{j}$ および $\mathbf{V}_{j}$ のサイズ $B_{c}\times d$ がオンチップメモリに収まる必要があり、これは次のように変換されます：

$$
B_{c}d=O(M)\Leftrightarrow B_{c}=O\left(\frac{M}{d}\right).
$$

同様に、サイズ $B_{r}\times d$ のブロック $\mathbf{Q}_{i},\mathbf{O}_{i}$ もオンチップメモリに収まる必要があり、これは次のように変換されます：

$$
B_{r}d=O(M)\Leftrightarrow B_{r}=O\left(\frac{M}{d}\right).
$$

最後に、ブロック $\mathbf{S}_{\mathrm{ij}}$ のサイズ $B_{r}\times B_{c}$ がオンチップメモリに収まる必要があります。これは次のように表されます：

$$
B_{r}B_{c}=O(M).
$$

したがって、次のように設定します：

$$
B_{c}=\Theta\left(\frac{M}{d}\right),\qquad B_{r}=\Theta\left(\min\left(\frac{M}{d},\frac{M}{B_{c}}\right)\right)=\Theta\left(\min\left(\frac{M}{d},d\right)\right).
$$

次の式を得ます：

$$
T_{c}=\frac{N}{B_{c}}=\Theta\left(\frac{\mathrm{Nd}}{M}\right).
$$

その結果、HBM アクセスの回数は次の通りです：

$$
\Theta\left(\mathrm{NdT}_{c}\right)=\Theta\left(\frac{N^{2}d^{2}}{M}\right).
$$

∎

###### [命題3](#Thmtheorem3) の証明

反証のために、すべての $M\in[d,\mathrm{Nd}]$ に対して HBM アクセス数が次の通りである正確なアテンションを計算するアルゴリズムが存在すると仮定します：

$$
o\left(\frac{N^{2}d^{2}}{M}\right).
$$

$M=\Theta(\mathrm{Nd})$ の領域では、これにより HBM アクセス数は次のようになります：

$$
o\left(\frac{N^{2}d^{2}}{\mathrm{Nd}}\right)=o(\mathrm{Nd}).
$$

しかし、アテンションへの入力（行列 $\mathbf{Q},\mathbf{K},\mathbf{V}$）と出力 $\mathbf{O}$ のサイズは $\mathrm{Nd}$ であり、それらは最初から HBM 内にあるため、アルゴリズムが正確なアテンションを計算するなら、少なくとも $\Omega(\mathrm{Nd})$ 回の HBM アクセスが必要です。これは矛盾です。∎

###### [定理5](#Thmtheorem5) の証明。

AttentionのバックワードパスのIO複雑性は、AttentionのフォワードパスのIO複雑性と非常に似ています （[定理2](#Thmtheorem2)）。ここでは証明の概略を示します。

まず、標準アテンションの逆伝播パスの IO 複雑度を分析します。入力 $\mathbf{Q},\mathbf{K},\mathbf{V},\mathbf{\mathrm{dO}}\in\mathbb{R}^{N\times d}$ は HBM に存在し、アルゴリズムの最後に出力 $\mathbf{\mathrm{dQ}},\mathbf{\mathrm{dK}},\mathbf{\mathrm{dV}}\in\mathbb{R}^{N\times d}$ は HBM に書き込まれます。

標準アテンション逆伝播パスの各ステップで、HBM からサイズ $\mathrm{Nd}$ または $N^{2}$ の入力をロードする必要があり、サイズ $N^{2}$ または $\mathrm{Nd}$ の出力を HBM に書き込む必要があります。これにより $\Theta(\mathrm{Nd}+N^{2})$ 回の HBM アクセスが発生します。

次にFlashAttentionバックワードパスのIO複雑性を分析します。

は [定理 2](#Thmtheorem2) に類似しており、$\mathbf{K}$ と $\mathbf{V}$ の各要素は HBM から一度だけ読み込まれることがわかります。$\mathbf{\mathrm{dK}}$ と $\mathbf{\mathrm{dV}}$ の各要素は HBM に一度だけ書き込まれます。$T_{c}$ 回 $\mathbf{Q},\mathbf{O},\mathbf{\mathrm{dO}}$ 上でパスを行い、各パスで $\mathbf{Q},\mathbf{O},\mathbf{\mathrm{dO}}$ 全体を HBM に読み込みます。また、$T_{c}$ 回 $\mathbf{\mathrm{dQ}}$ 上でパスを行い、各パスで $\mathbf{\mathrm{dQ}}$ 全体を HBM から読み書きします。したがって、HBM アクセスの回数は $\Theta\left(\mathrm{Nd}+\mathrm{NdT}_{c}\right)=\Theta(\mathrm{NdT}_{c})$ です。

[定理2](#Thmtheorem2) の証明と同様に、ブロックサイズの制約は次の通りです：

$$
B_{c}=\Theta\left(\frac{M}{d}\right),\qquad B_{r}=\Theta\left(\min\left(\frac{M}{d},d\right)\right).
$$

次の式を得ます：

$$
T_{c}=\frac{N}{B_{c}}=\Theta\left(\frac{\mathrm{Nd}}{M}\right).
$$

その結果、HBM アクセスの回数は次の通りです：

$$
\Theta\left(\mathrm{NdT}_{c}\right)=\Theta\left(\frac{N^{2}d^{2}}{M}\right).
$$

∎

## 付録D 拡張の詳細

### D.1 ブロックスパースFlashAttention

我々は、[アルゴリズム5](#alg5)で完全なブロックスパースFlashAttentionアルゴリズムを説明します。このアルゴリズムは、ゼロブロックをスキップする点を除けば、[アルゴリズム2](#alg2)と同一です。

<span id="alg5"></span>

**アルゴリズム 5: ブロックスパースFlashAttention 順方向計算**

- **入力:** サイズ$M$のHBM上の行列$\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}$、ソフトマックススケーリング定数$\tau\in\mathbb{R}$、マスキング関数mask、ドロップアウト確率$p_{\mathrm{drop}}$、ブロックサイズ$B_{c}=\left\lceil\frac{M}{4d}\right\rceil,B_{r}=\min\left(\left\lceil\frac{M}{4d}\right\rceil,d\right)$、ブロックスパースマスク$M\in\{0,1\}^{N/B_{r}\times N/B_{c}}$。
- 疑似乱数生成器の状態${\cal R}$を初期化し、HBMに保存。
- HBM上の$\mathbf{O}=(0)_{N\times d}\in\mathbb{R}^{N\times d},\ell=(0)_{N}\in\mathbb{R}^{N},m=(-\infty)_{N}\in\mathbb{R}^{N}$を初期化。
- $\mathbf{Q}$ を $T_{r}=\left\lceil\frac{N}{B_{r}}\right\rceil$ ブロック $\mathbf{Q}_{1},\dots,\mathbf{Q}_{T_{r}}$ に分割し、それぞれのサイズは $B_{r}\times d$ とし、$\mathbf{K},\mathbf{V}$ を $T_{c}=\left\lceil\frac{N}{B_{c}}\right\rceil$ ブロック $\mathbf{K}_{1},\dots,\mathbf{K}_{T_{c}}$ および $\mathbf{V}_{1},\dots,\mathbf{V}_{T_{c}}$ に分割し、それぞれのサイズは $B_{c}\times d$ とする。
- $\mathbf{O}$ を $T_{r}$ ブロック $\mathbf{O}_{i},\dots,\mathbf{O}_{T_{r}}$ に分割し、それぞれのサイズは $B_{r}\times d$ とし、$\ell$ を $T_{r}$ ブロック $\ell_{i},\dots,\ell_{T_{r}}$ に分割し、それぞれのサイズは $B_{r}$ とし、$m$ を $T_{r}$ ブロック $m_{1},\dots,m_{T_{r}}$ に分割し、それぞれのサイズは $B_{r}$ とする。
- **$1\leq j\leq T_{c}$ に対して繰り返す:**
  - $\mathbf{K}_{j},\mathbf{V}_{j}$ を HBM からオンチップ SRAM にロードする。
  - **$1\leq i\leq T_{r}$ に対して繰り返す:**
    - **もし** $M_{\mathrm{ij}}\neq 0$ **なら:**
      - $\mathbf{Q}_{i},\mathbf{O}_{i},\ell_{i},m_{i}$ を HBM からオンチップ SRAM にロードする。
      - チップ上で $\mathbf{S}_{\mathrm{ij}}=\tau\mathbf{Q}_{i}\mathbf{K}_{j}^\top\in\mathbb{R}^{B_{r}\times B_{c}}$ を計算します。
      - チップ上で $\mathbf{S}_{\mathrm{ij}}^{\mathrm{masked}}=\mathrm{mask}(\mathbf{S}_{\mathrm{ij}})$ を計算します。
      - チップ上で、$\tilde{m}_{\mathrm{ij}}=\mathrm{rowmax}(\mathbf{S}_{\mathrm{ij}}^{\mathrm{masked}})\in\mathbb{R}^{B_{r}}$、$\tilde{\mathbf{P}}_{\mathrm{ij}}=\exp(\mathbf{S}_{\mathrm{ij}}^{\mathrm{masked}}-\tilde{m}_{\mathrm{ij}})\in\mathbb{R}^{B_{r}\times B_{c}}$（点ごとに）、$\tilde{\ell}_{\mathrm{ij}}=\mathrm{rowsum}(\tilde{\mathbf{P}}_{\mathrm{ij}})\in\mathbb{R}^{B_{r}}$ を計算します。
      - チップ上で、$m_{i}^{\mathrm{new}}=\max(m_{i},\tilde{m}_{\mathrm{ij}})\in\mathbb{R}^{B_{r}}$、$\ell_{i}^{\mathrm{new}}=e^{m_{i}-m_{i}^{\mathrm{new}}}\ell_{i}+e^{\tilde{m}_{\mathrm{ij}}-m_{i}^{\mathrm{new}}}\tilde{\ell}_{\mathrm{ij}}\in\mathbb{R}^{B_{r}}$ を計算します。
      - チップ上で、$\tilde{\mathbf{P}}_{\mathrm{ij}}^{\mathrm{dropped}}=\mathrm{dropout}(\tilde{\mathbf{P}}_{\mathrm{ij}},p_{\mathrm{drop}})$ を計算します。
      - $\mathbf{O}_{i}\leftarrow\mathrm{diag}(\ell_{i}^{\mathrm{new}})^{-1}(\mathrm{diag}(\ell_{i})e^{m_{i}-m_{i}^{\mathrm{new}}}\mathbf{O}_{i}+e^{\tilde{m}_{\mathrm{ij}}-m_{i}^{\mathrm{new}}}\tilde{\mathbf{P}}_{\mathrm{ij}}^{\mathrm{dropped}}\mathbf{V}_{j})$ を HBM に書き込みます。
      - $\ell_{i}\leftarrow\ell_{i}^{\mathrm{new}}$、$m_{i}\leftarrow m_{i}^{\mathrm{new}}$をHBMに書き込む。
- **返却:** $\mathbf{O},\ell,m,{\cal R}$を返す。

ブロックスパースFlashAttentionのIO複雑度を証明する。

###### [命題4](#Thmtheorem4) の証明です。

証明は、[定理2](#Thmtheorem2) の証明と非常に似ています。ブロックスパースの場合、非ゼロブロックに対応するブロックのみをロードすればよいことに注意してください。その結果、HBMアクセスの回数はブロックスパースマスクにおける非ゼロブロックの割合 $s$ に応じてスケーリングされます。しかし、$s$ の値が小さい場合でも、結果を $\mathbf{O}\in\mathbb{R}^{N\times d}$ に書き込む必要があります。したがって、HBMアクセスの回数は

$$
\Theta\left(\mathrm{Nd}+\frac{N^{2}d^{2}}{M}s\right).
$$

∎

### D.2 潜在的な拡張

ここでは、ディープラーニングのトレーニングを高速化するための IO 認識アプローチのいくつかの潜在的な拡張について議論します。

マルチGPUアテンション。大規模言語モデルは数百または数千のGPUで訓練され、通常、アテンション計算は同一ノード上の4〜8 GPUに分割されます [Shoeyb19]。これにより、別のレベルのメモリ階層が導入されます：GPUのSRAMやGPUのHBMに加えて、他のGPUのHBMも利用可能です。非常に長いシーケンスの場合、同一ノード上の異なるGPUは、異なるメモリ階層レベルの非対称性を考慮しながらアテンションを計算するために協力できます。

スパースMLP層。典型的な密なMLP層は計算に制約されており、メモリによる制約はありません。効率を改善するために、スパース重み行列を持つMLP層を使用できます [Daoa22]。しかし、多くのスパースMLP層はむしろメモリ制約を受けやすく、スピードアップはスパース性に比例しないことが多いです。私たちは、I/Oに配慮した実装がこの問題を軽減し、スパース性の利点を実現できると考えています。我々は、この方向での将来の研究に期待しており、大規模モデルの計算要件を削減し、実行時間を改善することを目指しています。

カーネル機械学習。FlashAttention における我々のアプローチは、$N\times N$ 注意マトリックスが低ランクマトリックス $\mathbf{Q}\mathbf{K}^{\top}$（ランクは $d\ll N$）の関数であるという事実に依存しています。その結果、入力を繰り返し $\mathbf{Q},\mathbf{K}$ 読み込み、必要な注意マトリックスのブロックを再計算することができ、HBM アクセスを大幅に削減できます。カーネル機械学習でも同様のシナリオが発生します：$N\times N$ カーネルマトリックス $\mathbf{K}$ の各要素 $K_{\mathrm{ij}}$ は、2 つのベクトル（サイズは $d\ll N$）の関数であり、2 つのデータポイント $x_{i}$ と $x_{j}$ の類似度を測定します。KeOps ライブラリ [Feydy20, Charli21] は、メモリの読み書きを減らすことでカーネル演算を高速化できることを示す成功例です。これにより、単に FLOPs を減らすだけでなく、IO を削減することに重点を置いたカーネル手法の動機になることを期待しています。

## 付録 E 完全な実験結果

### E.1 BERT

我々は、参照 MLPerf 1.1 実装の訓練手順とハイパーパラメータに従って BERT-large を訓練する。特に、学習率 3.75e-3 の LAMB オプティマイザを使用し、バッチサイズ 448 で最大 7100 ステップまで訓練する。検証精度（マスク言語モデル用）が目標の 72.0% に達した時点で訓練は停止し、ウォールクロックの実行時間を測定する。FP16 精度で Apex AMP（O2 最適化レベル）を使用して訓練する。

我々の結果を、MLPerf 1.1 に提出された Nvidia の報告トレーニング速度と比較する（[表 1](#table-01)）。

我々はMLPerf 1.1リファレンス実装で提供されている同じ訓練／検証データ分割を使用します。特に、Nvidiaのベースラインと同じ10,000の検証例で評価します。

我々は8$\times$A100-80GB GPUでモデルを訓練します。各訓練実行は16〜19分かかり、10回の実行結果を平均します。

### E.2 GPT-2

我々はHuggingface transformersライブラリおよびNvidiaのMegatron-LMリポジトリからのGPT-2標準実装を使用します。我々はMegatron-LMリポジトリの訓練レシピに従います。

我々は有効バッチサイズ512を使用し、利用可能なGPUメモリに収まるように勾配蓄積を使用します。AdamWオプティマイザを使用し、学習率はGPT-2 smallで6e-4、GPT-2 mediumで1.5e-4、重量減衰は0.1とします。すべてのモデルは同じハイパーパラメータで400Kステップ訓練されます。すべての実装は混合精度訓練（PyTorch AMP）で実行されます。

我々はGPT-2 BPEトークナイザーを使用してOpenwebtextデータセットを使用します。データセットの0.5％をランダムに検証セットとして選択し、残りを訓練セットとして使用します。この検証セットのランダム選択は一度だけ行われ、すべてのモデルは同じ検証セットで評価されます。

我々は8$\times$A100-40GB GPUでモデルを訓練し、実際の訓練時間を測定します。GPT-2 smallの訓練には2.7〜9.5日、GPT-2 mediumの訓練には6.9〜21.0日かかります（[表2](#table-02)）。

[図 4](#figure-04)（#figure-04）では、HuggingFaceの実装または我々のFlashAttention実装を使用して、GPT-2 small/mediumのトレーニング中の検証パープレキシティをプロットしています。FlashAttentionはベースライン実装と同じ挙動を示し、両実装の検証パープレキシティ曲線はほぼ重なっていることがわかります。

<span id="figure-04"></span>

![キャプションを参照](../../papers/flashattention/figure-04.png)

**図4.** 2つの実装を使用したGPT-2 small/mediumの検証パープレキシティ。FlashAttentionはHuggingFaceのベースライン実装と同じ検証曲線を示すことを確認しました。

##### 長文文書分類

MIMIC-IIIおよびECtHRについては、[Dai22]のハイパーパラメータに従います。

### E.3 LRAの詳細

Long-range arenaの論文[Repree20]、Long-range arenaリポジトリ（https://github.com/google-research/long-range-arena](https://github.com/google-research/long-range-arena）、およびNyströmformer再現[Xiong21]のハイパーパラメータに従います。ベースライン手法に配慮して、5つのタスクのいずれかにおいて任意のベースラインの性能を再現できなかった場合、そのベースラインの当該タスクにおけるより良い性能を[Repree20]または[Xiong21]から報告します。

ハイパーパラメータ調整後、ほぼすべてのアテンション手法は5つのLRAタスクすべてで類似した精度を達成しています。

我々は、Performer（混合精度で安定しない）およびLocal Attention（実装がFP16をサポートしていない）を除き、すべての手法を混合精度トレーニングで実行します。

全体のウォールクロック時間の高速化を計算するために、5つのタスクそれぞれのウォールクロック時間高速化の幾何平均を取ります。

##### Path-X

Path-XおよびPath-256については、Long-Range Arena論文のPathFinder-32実験のハイパーパラメータに従います。
[Repree20] 両方とも、まずPath-64でモデルを事前学習します。200エポック後のチェックポイントを取得し、その位置埋め込みをアップサンプリングします（位置埋め込みを空間的にグリッド状に複製します）、そしてダウンストリームタスクで200エポック微調整します。1エポックの線形ウォームアップと学習率のコサイン減衰を行います。Path-Xについては、検証精度に基づき最良のチェックポイントを取得し、さらに同じウォームアップと学習率で200エポック微調整します（これにより、Path-Xに対するFlashAttentionの精度が約4ポイント向上しますが、その後モデルは過学習を始めます）。

### E.4 Apex FMHAとの比較

我々は、我々の手法／実装をApex FMHA （[https://github.com/NVIDIA/apex/tree/master/apex/contrib/csrc/fmha](https://github.com/NVIDIA/apex/tree/master/apex/contrib/csrc/fmha]） と比較します。

私たちがこのプロジェクトを開始したとき、Apex FMHAは注目機構の中で最も高速な実装であり（私たちが知っている限りでは）、最大512の長さの短いシーケンス向けに最適化されていました。実際、Nvidia GPU上で動作するBERTトレーニングベンチマークのほとんどすべてのMLPerf提出物は、MLPerf 1.1 [Mattso20]時点でモデルコードにFMHAを使用しています。FMHAはBERTモデルを対象としているため、ヘッド次元64のみに対応しており、A100 GPUでのみ動作します。FMHAは注目計算$\mathrm{dropout}(\mathrm{softmax}(\mathrm{mask}(\mathbf{Q}\mathbf{K}^{\top})))\mathbf{V}$を1つのCUDAカーネルに統合します。順方向計算では、勾配計算で使用するために注目行列$\mathrm{softmax}(\mathrm{mask}(\mathbf{Q}\mathbf{K}^\top))$をHBMに格納します。その結果、大幅なメモリ節約は提供しません（ただし、短いシーケンスの場合、メモリの使用量はしばしば主要な問題ではありません）。

私たちはFMHAコードを出発点として使用し、長いシーケンスに対処し、[セクション3](#S3) で述べたようにメモリを節約するために、2つの確立された手法（タイル化と再計算）を適用しました。その結果、はるかに長いシーケンス（例えば最大長64K）をサポートできるようになりました。また、より多くのヘッド次元（16、32、64、128）とより広範なGPUタイプ（執筆時点のすべてのTuringおよびAmpere GPU）もサポートしています。

[表 7](#table-07) （#table-07） では、短いシーケンスにおける FlashAttention と Apex FMHA の性能を比較しています（FMHA はシーケンス長最大512のみサポートするため）。一般的に、FlashAttention は順伝播では FMHA よりわずかに速く、逆伝播では FMHA よりわずかに遅くなります。これは、順伝播ではアテンション行列を保存せず、逆伝播で再計算するためです。FMHA と比較すると、FlashAttention の総合的な実行時間は、シーケンス長128では約4%遅く、シーケンス長256では8%速く、シーケンス長512では5%速くなります。

<span id="table-07"></span>

![論文の表 7](../../papers/flashattention/table-07.png)

**表7.** FlashAttentionのランタイム（ミリ秒）を、シーケンス長ごとにFMHAと比較。マスキングとドロップアウトあり。測定はA100-SXM4-40GB GPUで実施。バッチサイズ64、ヘッド数16、ヘッド次元64（すなわちBERT-largeサイズ）。

### E.5 異なるハードウェアと構成での高速化

高速化は、HBM帯域幅やSRAMサイズに依存して、GPUの種類や世代によって異なります。本節では、異なるGPUと構成でのFlashAttentionの高速化をプロファイリングします。

<span id="figure-05"></span>

![キャプションを参照](../../papers/flashattention/figure-05.png)

**図5.** A100上での標準PyTorch Attentionに対する異なるシーケンス長での高速化。

##### A100

[図5](#figure-05)は、バッチサイズ8、ヘッド次元64、12個のAttentionヘッドでのA100 GPU上の異なるシーケンス長における高速化を示しています。一般的に2〜4倍の高速化が見られ、カーネルフュージョンによりドロップアウトやマスキングを使用するとさらに高速化が見られます。

<span id="figure-06"></span>

![キャプションを参照](../../papers/flashattention/figure-06.png)

**図6.** ヘッド次元128のA100上での標準PyTorch Attentionに対する異なるシーケンス長での高速化。

##### A100、ヘッド次元128

の速度向上も変わります。各ブロックはより多くのメモリを必要とするため、SRAMに収まるように小さいブロックサイズを使用する必要があります。[図6](#figure-06)は、A100上でヘッド次元128の速度向上を示しています（バッチサイズ16、ヘッド12）。全体的には速度向上は少なくなりますが、因果マスクを使用した場合（ブロックの半分がマスクされる）には、それでも最大3$\times$までの顕著な速度向上が見られます。

<span id="figure-07"></span>

![キャプションを参照](../../papers/flashattention/figure-07.png)

**図7.** RTX 3090上での異なるシーケンス長における標準PyTorchアテンションに対する速度向上。

##### RTX 3090

[図7](#figure-07)はRTX 3090 GPU上での速度向上を示しています。ここでは、バッチサイズ12、アテンションヘッド12を使用しています。RTX 3090では（2.5〜4.5$\times$の範囲で）若干高い速度向上が観察されます。これはRTX 3090のメモリ帯域幅がA100より低いためです（約900 GB/s対1.5 TB/s）。

<span id="figure-08"></span>

![キャプションを参照](../../papers/flashattention/figure-08a.png)

![キャプションを参照](../../papers/flashattention/figure-08b.png)

**図8.** T4上での異なるシーケンス長における標準PyTorchアテンションに対する速度向上。上：順伝播+逆伝播の合計。下：順伝播のみ。

##### T4

[図8](#figure-08) は T4 GPU におけるスピードアップを示しています。T4 の SRAM は A100 より小さいため、FlashAttention ではブロックサイズを小さくする必要があります。その結果、T4 ではスピードアップが小さくなることが観察され、これは [3.2節](#S3.SS2) の IO 複雑性分析と一致しています。T4 GPU は推論に一般的に使用されるため、フォワードパスのみのスピードアップも報告します。

### E.6 完全なベンチマーク結果

A100 における完全なベンチマーク結果と実験の詳細を報告します。

##### ベースライン

PyTorch/HuggingFace および Megatron の正確な Attention の参照実装、近似 Attention、およびスパース Attention と比較します。近似 Attention については、Reformer [ICML20]、Local Attention [July20]、Linformer Attention [Wang20]、Smyrf [Daras20]、LongShortFormer （LSFormer） [Systej21] の参照実装と比較します。スパース Attention については、OpenAI の Block-Sparse Attention、Longformer [Beltag20]、BigBird Attention [Systeo20] の参照実装と比較します。近似およびスパース Attention では、圧縮率 1/8、または圧縮後のシーケンス長 256 の小さい方を使用します。

##### 設定

我々は、次元64の8つのヘッドとバッチサイズ16での注意計算の実行時間とメモリ使用量を、40GBのGPU HBMを搭載したA100 GPU 1台のマシン上で測定します。実験ではシーケンス長を変化させます。$\mathbf{Q}$、$\mathbf{K}$、および$\mathbf{V}$では、ランダムベクトル上で注意計算を行います（隠れ層からの射影は測定しません）。ドロップアウトにはドロップアウト率0.1を使用し、マスキングには全シーケンス長から全シーケンス長マイナス20までの一様ランダムマスク長を使用したパディングマスクを使用します。実行時間を測定するために、注意呼び出しを100回測定した平均値を取ります。メモリ使用量は実行間で変動しないため、1回のみ測定します。

順伝播、逆伝播、および順伝播と逆伝播の合計に対するタイミング結果を報告します。各手法は、ドロップアウト、マスキング、またはその両方の有無で測定します。ただし、Block Sparse、Longformer、および BigBird は例外です。これらの手法は、外部ライブラリのバグによりマスキング付き逆伝播が正常に実行できなかったため、寛大にマスキングなしで測定しました。すべての測定にFP16を使用しますが、Local Attentionは実装上FP32のみをサポートしています。

各ベースラインについて、GPU上でメモリが不足するまでシーケンス長を増加させます。ただし、次の例外があります：Megatronの実装はシーケンス長2048を超えることをサポートしていません。Block-Sparse（OpenAI）はシーケンス長4096を超えることをサポートしていません。LongformerおよびBigBirdはシーケンス長8092を超えることをサポートしていません。

ドロップアウトやマスキングなしで、順伝播と逆伝播を組み合わせたパスでのメモリ使用量を測定します。

##### 結果

[表8](#table-08) はすべての実験設定を要約しており、結果表への参照を含んでいます。

<span id="table-08"></span>

![論文の表 8](../../papers/flashattention/table-08.png)
|はい|はい|結合済み|[表11](#table-11)
ZX0023QXZ  いいえ|
|いいえ|はい|前方|[表12](#table-12)|
|いいえ|はい|後方|[表13](#table-13)|
|いいえ|はい|結合済み|[表14](#table-14)|
|はい|いいえ|前方|[表15](#table-15)|
|はい|いいえ|後方|[表16](#table-16)|
|はい|いいえ|結合済み|[表17](#table-17)|
|いいえ|いいえ|前方|[表18](#table-18)|
|いいえ|いいえ|後方|[表19](#table-19)|
|いいえ|いいえ|結合済み|[表20](#table-20)|
|いいえ|いいえ|メモリ使用量（結合済み）|[表21](#table-21)|

**表8.** 結果表へのポインタ

<span id="table-09"></span>

![論文の表 9](../../papers/flashattention/table-09.png)

**表9.** ドロップアウトとマスキングありでの様々な正確／近似／スパースアテンション機構におけるシーケンス長ごとのフォワードパス実行時間（ms）。最良は太字、次点は下線。

<span id="table-10"></span>

![論文の表 10](../../papers/flashattention/table-10.png)

**表10.** 様々な正確/近似/スパース注意メカニズムにおけるシーケンス長別の逆伝播実行時間（ms）、ドロップアウトおよびマスキングあり。最良は太字、次点は下線。

<span id="table-11"></span>

![論文の表 11](../../papers/flashattention/table-11.png)

**表11.** 順伝播および逆伝播の実行時間（ms）、シーケンス長ごとの様々な正確／近似／スパース注意機構（ドロップアウトとマスキング付き）。最高は太字、2番目は下線。

<span id="table-12"></span>

![論文の表 12](../../papers/flashattention/table-12.png)

**表12.** マスキングありの各種正確/近似/スパース注意機構のシーケンス長別フォワードパス実行時間（ms）。最良は太字、次点は下線。

<span id="table-13"></span>

![論文の表 13](../../papers/flashattention/table-13.png)

**表13.** マスキング付きの様々な正確/近似/スパース注意メカニズムの順伝播計算時間（ミリ秒）、シーケンス長ごと。最良は太字、次に良いものは下線付き。

<span id="table-14"></span>

![論文の表 14](../../papers/flashattention/table-14.png)

**表14.** マスキング付きのさまざまな正確/近似/スパースアテンション機構のシーケンス長ごとの順伝播および逆伝播ランタイム（ms）。最良は太字、2番目に良いものは下線。

<span id="table-15"></span>

![論文の表 15](../../papers/flashattention/table-15.png)

**表15.** ドロップアウトありの各種正確/近似/スパース注意メカニズムのシーケンス長ごとのフォワードパス実行時間（ms）。最良は太字、2番目に良いものは下線。

<span id="table-16"></span>

![論文の表 16](../../papers/flashattention/table-16.png)

**表16.** ドロップアウトありのシーケンス長ごとのさまざまな正確/近似/スパース注意メカニズムのバックワードパス実行時間（ms）。最良は太字、2番目に良いは下線。

<span id="table-17"></span>

![論文の表 17](../../papers/flashattention/table-17.png)
|ローカルアテンション|1.09|1.40|1.99|5.61
ZX0032QXZ  19.23| 19.23 |38.62|77.30|154.63|311.12|-|
|Linformer|1.31|1.21|1.30|1.39|3.73|7.15|14.05|27.69|55.00|-|
|Smyrf|3.00|4.37|8.05|15.66|31.04|61.64|123.04|245.65|-|-|
|LSformer|3.07|3.17|4.31|10.89|31.54|61.78|121.56|240.94|-|-|
|Block Sparse|2.54|2.52|3.71|5.44|13.29|39.19|-|-|-|-|
|Longformer|2.47|2.49|2.51|3.10|10.39|22.49|60.44|-|-|-|
|BigBird|2.51|2.49|2.52|3.40|10.97|23.89|63.28|-|-|-|
|FlashAttention|0.35|0.36|0.80|2.52|9.16|36.70|146.13|583.45|2332.01|9323.63|
|ブロックスパースフラッシュアテンション|0.91|0.83|0.94|0.92|1.83|3.50|7.02|13.56|26.71|53.92|

**表17.** ドロップアウトありでの様々な正確/近似/スパースアテンション機構のシーケンス長ごとの順伝播・逆伝播実行時間（ms）。最良は太字、次点は下線。

<span id="table-18"></span>

![論文の表 18](../../papers/flashattention/table-18.png)

**表18.** さまざまな正確/近似/スパース注意機構のシーケンス長別フォワードパス実行時間（ms）。最良は太字、次点は下線付き。

<span id="table-19"></span>

![論文の表 19](../../papers/flashattention/table-19.png)
|PyTorchアテンション|0.26| 0.29 |0.78|2.44
ZX0031QXZ  8.82
ZX0032QXZ  33.87
ZX0033QXZ  -
ZX0034QXZ  -
ZX0035QXZ  -
ZX0036QXZ  -
ZX0037QXZ  Megatron
ZX0038QXZ  0.29
ZX0039QXZ  0.30
ZX0040QXZ  0.80
ZX0041QXZ  2.59
ZX0042QXZ  8.86
ZX0043QXZ  -
ZX0044QXZ  -
ZX0045QXZ  -
ZX0046QXZ  -| 8.82 | 33.87 |-|-|-|-|
|Megatron| 0.29 | 0.30 |0.80| 2.59 | 8.86 |-|-|-|-|-|
|Reformer|2.18|4.21|8.14|16.12|32.02|63.84|127.60|-|-|-|
|ローカルアテンション|0.51|0.64|1.28|3.60|12.52|25.08|50.22|100.23|200.66|-|
|Linformer|0.69|0.76|0.69|0.80|2.04|3.88|7.67|15.04|30.11|63.15|
|Smyrf|1.24|2.49|4.77|9.42|18.65|37.12|74.15|148.35|-|-|
|LSformer|1.68|1.61|3.02|7.40|19.72|38.27|74.89|147.99|-|-|
|Block Sparse|1.24|1.25|2.04|2.91|6.78|19.67|-|-|-|-|
|Longformer|1.27|1.23|1.24|1.85|4.99|10.21|24.89|-|-|-|
|BigBird|1.43|1.50|1.44|1.69|5.25|10.86|26.26|-|-|-|
|FlashAttention|0.11|0.16|0.52|1.62|5.45|21.57|84.75|336.00|1338.56|5343.19|
|ブロックスパースフラッシュアテンション|0.11|0.12|0.16|0.38|1.20|2.34|4.69|9.10|18.74|37.04|

**表19.** シーケンス長ごとのさまざまな正確/近似/スパース注意機構の逆伝播実行時間（ms）。最良は太字、次に良いものは下線。

<span id="table-20"></span>

![論文の表 20](../../papers/flashattention/table-20.png)

**表20.** さまざまな正確/近似/スパース注意メカニズムのシーケンス長ごとの順伝播・逆伝播実行時間（ms）。最良は太字、次点は下線。

<span id="table-21"></span>

![論文の表 21](../../papers/flashattention/table-21.png)

**表21.** シーケンス長ごとの様々な正確/近似/スパースアテンション機構のメモリ使用量（MB）。最高は太字、2番目は下線付き。

[+1]： FlashAttention のコードは [https://github.com/HazyResearch/flash-attention](https://github.com/HazyResearch/flash-attention] で利用可能です。

[+2]： このスタイルの集約は *代数的集約* と呼ばれます [Gray97]。

[+3]： LRA の精度結果はチューニング手順に大きく依存することが知られています [Xiong21]。再現したベースラインは元の比較で報告されたものよりも優れた性能を示しています [Repree20]。

[+4]： Path-256 はより長いシーケンスを必要としますが、Path-X よりも比較的短いパスを持つため、高い精度を得やすいです。
