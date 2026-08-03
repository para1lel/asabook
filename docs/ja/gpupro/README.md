---
title: "機械学習システムのための現代 GPU プログラミング"
createTime: 2026/08/01 00:00:00
permalink: /ja/gpupro/
pageClass: gpupro-page
---

::: note 出典
このローカル版は [MLC Community の原書](https://mlc.ai/modern-gpu-programming-for-mlsys/)を[上流コミット `a5ed072f0d35`](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys/tree/a5ed072f0d35c35722bbe86dec6926baad2aa46a)から取り込み、日本語に翻訳したものです。Copyright 2026 MLC Community。本文とアセットには、翻訳および VuePress に必要な書式・リンクの調整以外の変更を加えていません。上流リポジトリには現在、リポジトリ全体を対象とするライセンスファイルがありません。TIRx リファレンスページには Apache License 2.0 の表記を残しています。
:::

機械学習システムは、現代 AI の中核的な計算タスクをサポートします。 モデルスケールが拡大し、展開シナリオが複雑になるにつれて、システムのパフォーマンスはいくつかの主要な GPU kernel の実装品質にますます依存しています。 attention kernel、LLM プリフィルおよびデコード kernel、低精度ブロックスケール GEMM、融合 MoE レイヤー、その他の大規模な融合 kernel はすべて、エンドツーエンドのトレーニングおよびサービス速度に直接影響します。

したがって、現代の AI システムを理解し最適化するには、高性能 GPU kernel がどのように書かれているかを理解することが不可欠です。 しかし、高性能 kernel は単にスタッキング最適化技術の結果ではありません。 現代の GPU アーキテクチャは大きな変化を遂げており、新しいアーキテクチャにより豊かなメモリ空間、新しいデータ処理機構、そしてますます専門化される実行ユニットが導入されています。 これらのハードウェア能力を最大限に活用するためには、明確なハードウェアのメンタルモデルを構築し、高性能 kernel がベースバージョンから段階的に進化する仕組みを理解する必要があります。 本書はまさにこの二つの側面に焦点を当てています。

この目標に基づき、本書はハードウェアからコード、そして高性能 kernel へと順に提供されます。 まず GPU のハードウェアの組織化と実行モデルを紹介し、次に本書で用いられているプログラミングモデルを学び、最後にこれらの基盤に基づいて段階的に高度な GPU kernel を構築していきます。 具体的には、本書では NVIDIA Blackwell アーキテクチャを例として用いて、一般行列行列乗算(GEMM)と FlashAttention を詳細に説明します。 これらの kernel 構築中、データ layout、非同期データ処理、非同期コラボレーションといった GPU 最適化の重要なトピックも体系的に学びます。

本書の内容はカーネギーメロン大学の[Machine Learning Systems](https://mlsyscourse.org/)コースシリーズから派生しています。 これらの概念を学習可能にし、実行し、実際のコードで検証できるようにするために、本書は TIRx Python DSL を用いて GPU kernel の例をステップバイステップで行っています。 TIRx はハードウェアに近い構造で、基礎となる実行抽象化を公開するため、リーダーは制御フロー、メモリアクセス、同期ロジックを考えながらコードを実行できます。

この本はオープンソースプロジェクトです。貢献、改訂、例は[GitHub リポジトリを通じて](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys)歓迎します。


## この本の構成

- パート 1：GPU の理解。この節では、GPU の全体的なアーキテクチャと構成、高性能 kernel を書く一般的なアプローチ、データ layout、非同期メモリ操作や協働などの重要な概念を紹介し、以下の章で依存するハードウェアの理解を確立します。
- パート 2：TIRx の概要。このセクションでは、TIRx のコアコンポーネントを紹介し、次の章のコード例を理解する準備をします。
- パート 3：GEMM：tile から SOTA へ。このセクションでは、tile 化された GEMM を最適化する方法を包括的に説明し、TMA パイプライン化、永続的スケジューリング、warp specialization、2-CTA cluster を徐々に取り入れていきます。
- パート 4：Flash Attention 4。このパートはパート 3 の技術を基にした完全な attention kernel を構築します: 2 つの MMA、中央に挿入された softmax、オンライン softmax のリスケーリング、causal masking、GQA を含みます。
- 参考文献。TIRx 言語の参考文献、内部コンパイラ機構、非同期 kernel デバッグガイド。

## 目次

### 第一部: 理解 GPU

- [GPU 実行モデル](./gpu-execution-model/)
- [高性能 kernel の要点](./kernel-performance/)
- [データ layout とその表記法](./data-layout/)
- [Tensor Core データ layout の進化](./tensor-core-data-layouts/)
- [非同期データ転送: TMA](./tma/)
- [Blackwell Tensor Core: tcgen05.mma](./blackwell-tensor-core/)
- [Tensor Memory (TMEM)](./tensor-memory/)
- [非同期協調: mbarrier](./mbarrier/)
- [高度なスケジューリング: Cluster Launch Control](./cluster-launch-control/)

### パート 2: TIRx の概要

- [TIRx 入門](./tirx-introduction/)
- [TIRx Layout API](./tirx-layout-api/)

### パート 3: TILED から SOTA までの GEMM

- [Tiled GEMM の構築](./tiled-gemm/)
- [TMA による GEMM のパイプライン化](./pipelined-gemm/)
- [Warp Specialization と Cluster による GEMM のスケーリング](./warp-specialized-gemm/)

### パート 4: Flash Attention 4

- [Flash Attention 4](./flash-attention-4/)

### 参考文献

- [リファレンス](./reference/)
- [TIRx 言語リファレンス](./tirx-language-reference/)
- [Warp-Specialized Kernel のデバッグ](./debugging-warp-specialized-kernels/)
- [コンパイラ内部](./compiler-internals/)
