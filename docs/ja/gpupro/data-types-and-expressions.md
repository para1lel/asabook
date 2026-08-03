---
title: "データ型と式"
createTime: 2026/08/01 00:00:00
permalink: /ja/gpupro/data-types-and-expressions/
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

各 TIRx 式は低レベルの d 型と高レベルの型を持ちます。

## d 型

`PrimExpr` `.dtype` はスカラーやベクトルの要素の種類を表し、例えば `float32`、 `float16`、 `bfloat16`、 `int32`、 `uint8`、 `bool`、低精度 `float8_e4m3fn` / `float4_e2m1fn` はポインタの `handle` を表し、 `float32x4` のようなベクトル型も含まれます。 CUDA を生成する際、各 dtype は対応する CUDA タイプとなります。 以下では、複数の種類の dtype ローカルバッファと共有バッファを同時に割り当て、 `float32x4` ベクターロード/ストアを一度実行します。

```python
@T.prim_func
def dtypes(A_ptr: T.handle, O_ptr: T.handle):
  A = T.match_buffer(A_ptr, (256,), "float32")
  O = T.match_buffer(O_ptr, (256,), "float32")
  T.device_entry(); bx = T.cta_id([1]); tx = T.thread_id([64])
  f16  = T.alloc_local((1,), "float16")        # register scalars ...
  bf16 = T.alloc_local((1,), "bfloat16")
  i32  = T.alloc_local((1,), "int32")
  u8   = T.alloc_local((1,), "uint8")
  b1   = T.alloc_local((1,), "bool")
  sm   = T.alloc_shared((64,), "float16")      # ... and a shared tile
  # A vector-dtype register (float4).
  v = T.alloc_local((1,), "float32x4")
  v[0] = A.vload([tx * 4], dtype="float32x4")  # vectorized load
  O.vstore([tx * 4], v[0])                     # vectorized store
  # ... (use f16/bf16/i32/u8/b1/sm) ...
```

生成される CUDA は、無関係なコードを省略した以下の通りです。

```c++
half          f16_ptr[1];               // float16
nv_bfloat16   bf16_ptr[1];              // bfloat16
int           i32_ptr[1];               // int32
uchar         u8_ptr[1];                // uint8
signed char   b1_ptr[1];                // bool
__shared__ alignas(64) half sm_ptr[64]; // shared float16
float4        v_ptr[1];                 // float32x4  (vector)
v_ptr[0]                  = *(float4*)(A_ptr + tx * 4);   // vectorized load
*(float4*)(O_ptr + tx * 4) = v_ptr[0];                   // vectorized store
```

バッファ自体もベクトル dtype を使うことができます。 `T.alloc_local((1,), "float32x4")` は直接 `float4` レジスタを宣言し、 `v[0]` を通じてアクセスされます。 `float32x4` `vload` / `vstore` は 16 バイトのアクセスで 1 つだけ伝送されます。 ベクトル dtype は `vload` だけでなく、通常のバッファやスカラーも使用可能です。

dtype 型と CUDA 型の共通対応は以下の通りです。

| dtype → CUDA | dtype → CUDA | dtype → CUDA |
| --- | --- | --- |
| `float32` → `float` | `float16` → `half` | `bfloat16` → `nv_bfloat16` |
| `int32` → `int` | `uint8` → `uchar` | `bool` → `signed char` |
| `float32x4` → `float4` | `handle` → `T*` (ポインター) | ベクトル dtype → CUDA ベクトル型 |

## dtype と type

`dtype` は基礎となる表現であり、値がどのビットから構成されるかを表します。 もう一つの値はより高次の型で、スカラーは `PrimType(dtype)`、ポインタは `PointerType(PrimType(dtype), scope)` を使用します。 ほとんどの式はスカラー(`PrimType`)です。 型システムは主にポインタを扱う際に役割します。

## ポインター(`handle`)

バッファの `data` はポインタであり、ポインタタイプの `Var` でもあります。 それは不変のであり、再割り当てはできません。 したがって、ポインタを取得する方法は 3 つあります。

- `T.alloc_buffer(...)` ストレージを割り当て、その `data` ポインタを定義します。
- `T.decl_buffer(..., data=ptr)` 既存のポインタ `Var` `ptr` にバッファを宣言します。
- 例えば、 `T.ptx.map_shared_rank` (PTX `mapa`)を使って他のクラスタ CTA の共有アドレスを取得するなど、ポインタ式付きのバッファをサポートしたい場合は、まずその式を `PointerType` の `T.let` を通じてポインタにバインドする必要があります `Var`。 `data` は `Var` でなければならず、式は直接使ってはいけません。

```python
from tvm.ir.type import PointerType, PrimType

ptr: T.let[T.Var(name="ptr", dtype=PointerType(PrimType("uint64")))] = \
  T.reinterpret("handle", T.ptx.map_shared_rank(mbar.ptr_to([0]), 0))
remote_mbar = T.decl_buffer([1], "uint64", data=ptr, scope="shared")
```
