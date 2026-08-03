---
title: "TIRx Lowering パイプライン"
createTime: 2026/08/01 00:00:00
permalink: /ja/gpupro/tirx-lowering-pipeline/
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

`tvm.compile(mod, target, tir_pipeline="tirx")` を呼び出すと、コンパイラは入力 TIRx モジュールに一連の TIR パスを順次送信します。これを TIR パイプラインと呼びます。 これは、tile プリミティブを徐々に減らし、 `TileLayout` バッファや実行 scope ID をホストやデバイス関数分離した機能に使い、最終的には CUDA のバックエンドはソースコードを生成します。

パイプラインは `python/tvm/tirx/compilation_pipeline.py` の `tirx_pipeline` で定義されています。以下では、実行順にパスを紹介します。

## コンパイルフローにおけるパイプラインの位置

`tvm.compile` まずターゲットをバインドし、その後、モジュールレベルの tirx パイプラインを実行します。 その後、ホスト関数とデバイス関数がそれぞれ最終化を通過し、デバイス機能は最終的に CUDA コードジェネレーターに引き継がれます。

```text
authored TIRx
  │
  └─BindTarget─▶ tirx_pipeline
                   ├─▶ host func ──host finalize──▶ C/LLVM
                   └─▶ device func ─device finalize─▶ CUDA
```

## 実行順序のパス

`tirx_pipeline` 以下の表のパスを順番に実行し、いくつかのパスは構成 `PassContext` 制御して実行します。

| # | パス | 機能 |
| --- | --- | --- |
| 1 | `LowerTIRx` | TIRx のコア降下完了は、以下の[Lower 内部](#lowertirx-は内部で何をしているのですか)で確認できます |
| 2 | `UnifyThreadBinding` | 等価なスレッド軸バインディングをマージし、各 `threadIdx` / `blockIdx` 軸は一度だけ宣言されるようにします |
| 3 | `StmtSimplify` | Arithmetic Analyzer を使って、文の算術式を簡略化してください |
| 4 | `LowerTIRxOpaque` | 残りの不透明な TIRx 構造体を標準 TIR に下げる |
| 5 | `FlattenBuffer` | 多次元 `BufferLoad` / `BufferStore` を一次元的なアクセスに平坦化する |
| 6 | `BF16ComputeLegalize` | 計算 `bfloat16` 法的な形に書き直し、計算は f32 に昇格します |
| 7 | `NarrowDataType(32)` | 安全性が証明されると、インデックスとループの `PrimExpr` d 型は 32 ビットに絞られます |
| 8 | `VectorizeLoop` | ループ `T.vectorized` ベクトル演算として書き換え; 設定時はスキップ `tir.disable_vectorize` |
| 9 | `UnrollLoop` | `T.unroll` とマークされた拡張ループや、より小さな定数ループも同様に有効です |
| 10 | `StmtSimplify` | ベクトル化してアンロールでさらに定数を露出した後、再び簡略化が行われます |
| 11 | `CommonSubexprElim` | 重複部分式を一時変数として抽出する; 設定時はスキップ `tir.disable_cse_tir` |
| 12 | `FP8ComputeLegalize` | 計算 `float8` 法的な形に書き換えましょう |
| 13 | `VerifyMemory` | ホストコードが直接デバイスメモリをデリファレンスしていないか確認してください |
| 14 | `AnnotateEntryFunc` | モジュール内の一意な PrimFunc をエントリ関数としてタグ付けします |
| 15 | `SplitHostDevice` | `launch_thread` 境界で各 kernel をホスト関数とデバイス関数分割します |
| 16 | `MakePackedAPI` | ホスト関数を TVM ランチャーで使われるパック関数 ABI に書き換えました |
| 17 | `FP8StorageLegalize` | バックエンドがサポートするコンテナタイプにストレージ `float8` パッケージ化します |
| 18 | `BF16StorageLegalize` | `bfloat16` ストレージを正当な形に書き換えましょう |

その後、コンパイラは関数型に従って最終化を実行します。

- ホスト: `LowerTVMBuiltin` `tvm_*` ビルディンを下げ、ターゲット固有の内在性を下げ `LowerIntrin`。
- デバイス: `LowerWarpMemory` warpscope バッファをシャッフルに降ろし、その後 `StmtSimplify` と `LowerIntrin` を実行します。

## LowerTIRx は内部で何をしているのですか?

`LowerTIRx` 自体は、 `src/tirx/transform/lower_tirx.cc` で定義される 2 つのパスから成り立っています。

```text
LowerTIRx = Sequential([ TilePrimitiveDispatch, LowerTIRxCleanup ])
```

- `TilePrimitiveDispatch` 選択したバックエンド dispatch に基づき、各 `TilePrimitiveCall` (`copy`、 `gemm`、 `reduction` など)を対応する実装に置き換えます。
- `LowerTIRxCleanup`` LayoutApplier ` を実行し、 ` TileLayout ` を使ったバッファアクセスを実際の物理アドレス計算(` addr = data + elem_offset + layout.apply(coord) `)に変換し、バッファをフラット化し、実行範囲 ID をスレッドに低く設定します 例えば軸は ` T.cta_id ` / ` T.thread_id ` ` launch_thread ` を通じて ` blockIdx ` / ` threadIdx` になります。

`LowerTIRx` を完了した後、モジュール内には通常の TIR: tile プリミティブのみが残り、 `TileLayout` 間接層は消え、scope ID はスレッド軸に解析されています。

## 完全な例

以下のスケール核を例として挙げます:

```python
@T.prim_func
def scale(A_ptr: T.handle, B_ptr: T.handle):
  A = T.match_buffer(A_ptr, (256,), "float32")
  B = T.match_buffer(B_ptr, (256,), "float32")
  T.device_entry(); bx = T.cta_id([1]); tx = T.thread_id([256])
  B[tx] = A[tx] * T.float32(2.0)
```

`LowerTIRx` を実行すると、scope ID は実のスレッド軸となり、layout はバッファアクセスに適用されます。ここで、 `A_1` と `B_1` は平坦化された一次元ビューです:

```python
with T.launch_thread("blockIdx.x", 1) as blockIdx_x:
  threadIdx_x = T.launch_thread("threadIdx.x", 256)
  bx: T.let = blockIdx_x
  tx: T.let = threadIdx_x
  B_1[threadIdx_x] = A_1[threadIdx_x] * T.float32(2.0)
```

`SplitHostDevice` と `MakePackedAPI` の後、関数はホストランチャーとデバイス kernel に分割されます。

```python
@I.ir_module
class Module:
  # Host packed-API launcher: computes the grid/block and launches.
  def main(...):
    ...
  def scale_kernel(...):  # device: the __global__ body, run on the GPU
```

CUDA バックエンドはその後 `__global__` 関数 `B_ptr[threadIdx.x] = A_ptr[threadIdx.x] * 2.0f` `scale_kernel` を生成します。

## 中間の結果を手動で確認してください

パイプラインのどのプレフィックスでも手動で実行して、ある段階で IR を確認できます。 本書の赤外線断片も次のように生成されます。

```python
from tvm.tirx import transform as TT

target = tvm.target.Target("cuda")
mod = TT.BindTarget(target.with_host("llvm"))(tvm.IRModule({"main": scale}))
# Dispatch tile primitives and apply layouts.
mod = TT.LowerTIRx()(mod)
print(mod.script())               # inspect the lowered TIRx IR
```

モジュール全体をコンパイルしてから生成された CUDA を確認することもできます:

```python
exe = tvm.compile(
  tvm.IRModule({"main": scale}),
  target=target,
  tir_pipeline="tirx",
)
print(exe.mod.imports[0].inspect_source())
```
