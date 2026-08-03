---
title: "パーサーユーティリティ"
createTime: 2026/08/01 00:00:00
permalink: /ja/gpupro/parser-utilities/
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

以下のツールは、TVMScript を TIRx に変換する解析時間適用されます。 Python 生成の値を直接 IR に書き込んだり、再利用可能なコードのスニペットを抽出したり、パーサーphase で使用された状態を整理したりできます。

## `T.meta_var`: インライン Python の値

`T.meta_var(x)` パーサーに、 `x` は Python によって計算されるコンパイル時のメタ値であり、スクリプト変数として解析されるのではなく、IR に直接組み込むべきだと伝えています。 実用的な用途のない一時変数を排除でき、メタプログラミングにも利用可能です。通常の Python ではメタ値を範囲として使用 `for` 解析時間中に展開します。

```python
n = T.meta_var(4)              # n is a Python int, inlined
for j in range(n):            # unrolled at parse time
  acc[0] = acc[0] + A[tx, j]
```

## `@T.inline`: インライン機能

`@T.inline` 定義関数は解析時間中に各呼び出し位置に組み込まれ、生成されたコードは関数呼び出しを保持しません。 これは Python の語彙(LEGB)scope とレイトバインディングに従っています。 関数パラメータは同名の外層変数を上書きします:

```python
@T.inline
def add_into(acc, x):
  acc[0] = acc[0] + x

add_into(acc, A[tx, j])       # inlined -> acc[0] = acc[0] + A[tx, j]
```

## `@T.meta_class`: パーサーの状態を整理する

`@T.meta_class` 通常の Python クラスを修正し、そのインスタンスをパーサのメタ値にします。 オブジェクトフィールドはバッファやスカラーを保存できるので、関連する割り当てと状態を同じオブジェクトに入れて kernel ボディで使うことができます。

```python
@T.meta_class
class State:
  def __init__(self, smem):
    self.acc = T.alloc_local([1], "float32")
    self.buf = T.decl_buffer([64], "float16", smem, scope="shared.dyn")

s = State(smem.data)
s.acc[0] = T.float32(0.0)     # use its fields like ordinary buffers
# ... s.buf[i] ...
```

この手法は、barrier、accumulator、一時ビューなどの kernel パイプライン状態の整理に適しており、関数内で多数の散発したローカル変数を渡すのを避けられます。

## `T.constexpr`

`T.constexpr` コンパイル時の kernel パラメータを宣言し、その値は `@T.jit` `.specialize(...)` で固定されます。 具体的な使用については、[TIRx 入門](/ja/gpupro/tirx-introduction/)を参照してください。
