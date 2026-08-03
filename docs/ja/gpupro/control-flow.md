---
title: "制御フロー"
createTime: 2026/08/01 00:00:00
permalink: /ja/gpupro/control-flow/
pageClass: gpupro-page
---

<!--
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements. See the NOTICE file
distributed with this work for additional information
regarding copyright ownership. The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied. See the License for the
specific language governing permissions and limitations
under the License.
-->

TIRx の制御ストリームには `if`、各種ループ、 `while` が含まれ、対応する CUDA 制御フローを生成します。

もし――

Python の `if` / `else` は CUDA の `if` / `else` になります。 糸やレーンの状態に基づいて作業区間を制限したり、 `T.ptx.elect_sync()` を使って経糸から糸を選び、指示を出すこともできます。

```python
if tx < 128:
  A[tx] = A[tx] * T.float32(2.0)
else:
  A[tx] = A[tx] + T.float32(1.0)

if T.ptx.elect_sync():
  ...                              # one elected lane (e.g. to issue TMA/MMA)
```

```c++
if (((int)threadIdx.x) < 128) {
  A_ptr[tx] = A_ptr[tx] * 2.0f;
} else {
  A_ptr[tx] = A_ptr[tx] + 1.0f;
}
```

式の値だけを選択してストリームの分岐を制御したい場合は `T.if_then_else(cond, a, b)` を使えます。 三進式に降格し、制御流発散を生じません。

```c++
O_ptr[tx] = (A_ptr[tx] > 0.0f) ? A_ptr[tx] : 0.0f;
```

## 一様制御フローと発散制御フロー

`if tx < 128` このようなスレッドごとのガードは通常の作業にも使用できますが、同期範囲内のすべてのスレッド一貫して共有を届けなければなりません。

例えば、 `T.cuda.cta_sync()` は `__syncthreads()` に対応し、スレッドブロック内のすべてのスレッドが参加する必要があります。 スレッド発散や warpgroup 発散分岐には置けません。 `if wg_id == 0:` に配置され、他の warpgroup が到達できない場合、kernel はデッドロックになります。 1 つの warpgroup のみを同期させる必要がある場合は、warpgroupscope 付き `T.cuda.warpgroup_sync(id)` を使用すべきです。warp 専門 GEMM および[CUDA C++/PTX 組み込み関数](/ja/gpupro/cuda-ptx-intrinsics/)のパート 3 を参照してください。

barrier を設置する際は、参加範囲に注意を払いましょう。 `mbarrier` `.init()` はシングルスレッドガード(`if (threadIdx.x < 1)`)に下げられます。 もし別の分岐に置くと、barrier が初期化されず、原因不明の発射失敗につながる可能性があります。

## ループ

TIRx は 4 種類のループを提供しています。 通常の Python `range` は次のように `T.serial` します:

- `T.serial(n)`: シーケンシャルループの場合、ptxas は拡張可能です。
- `T.unroll(n)`: 連続文に完全に拡張。
- `T.vectorized(n)`: ベクトル化されたループ。
- `T.grid(*extents)`: 多層の入れ子ループ。

Loop で `break` と `continue` を使えます。

```python
for i, j in T.grid(8, 8):
  B[i, j] = T.max(A[i, j], T.float32(0.0))
```

```c++
for (int i = 0; i < 8; ++i)
  for (int j = 0; j < 8; ++j)
    B_ptr[i * 8 + j] = max(A_ptr[i * 8 + j], 0.0f);
```

`T.unroll(4)` ループを生成するのではなく、直接 4 つの連続した文に展開します。

## ただし

`while` は条件が偽になるまで実行を続けます。 カウンターは可変スカラーを必要とします。詳細は[Buffer とメモリ](/ja/gpupro/buffers-and-memory/)を参照してください。

```python
i: T.int32 = 0
while i < 64:
  A[i] = A[i] + T.float32(1.0)
  i += 1
```

早期の退出 `break` で、 `while (1)` を下げることができます。 カウンターは 1 つの要素のみを持つレジスタバッファを使用します:

```c++
int i_ptr[1];
i_ptr[0] = 0;
while (1) {
  if (!(i_ptr[0] < 64)) { break; }
  A_ptr[i_ptr[0]] = A_ptr[i_ptr[0]] + 1.0f;
  i_ptr[0] = i_ptr[0] + 1;
}
```
