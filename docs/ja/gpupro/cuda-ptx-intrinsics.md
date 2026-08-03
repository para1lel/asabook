---
title: "CUDA C++/PTX 組み込み関数"
createTime: 2026/08/01 00:00:00
permalink: /ja/gpupro/cuda-ptx-intrinsics/
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

既存の tile プリミティブが必要な操作を表現できない場合、ハードウェアは 2 つの方法で直接アクセスできます。すなわち、 `tvm.backend.cuda` が提供する `T.cuda.*` / `T.ptx.*` 名前空間であるバックエンドの本質を呼び出すこと、 あるいは直接インラインの CUDA ソースコードも使えます。

## Call Backend Intrinsic

`T.cuda.*` および `T.ptx.*` は、CUDA バックエンドのデバイス内在要素、例えば同期、mbarrier、リダクション、PTX データ移動、MMA 命令を直接公開します。

```python
T.cuda.cta_sync()                    # block barrier (__syncthreads)
T.cuda.warp_sync()                   # __syncwarp
T.cuda.warpgroup_sync(8)             # warpgroup barrier
T.cuda.cta_sum(val, num_warps, scratch.ptr_to([0]))   # block-level reduction

bar = T.alloc_shared((1,), "uint64")
T.ptx.mbarrier.init(bar.data, 1)     # mbarrier for async completion
T.ptx.mbarrier.try_wait(bar.data, phase)
```

以下は `T.tvm_warp_shuffle_xor` を用いて warp オールリデュースを完了する完全な例です:

```python
@T.prim_func
def warp_reduce(A_ptr: T.handle):
  A = T.match_buffer(A_ptr, (32,), "float32", align=16)
  T.device_entry()
  cta_id = T.cta_id([1]); warp_id = T.warp_id([1]); lane_id = T.lane_id([32])
  v = T.alloc_local((1,), "float32"); i = T.alloc_local((1,), "int32")
  v[0] = T.float32(31 - lane_id)
  i[0] = 16
  while i[0] >= 1:
    v[0] += T.tvm_warp_shuffle_xor(0xFFFFFFFF, v[0], i[0], 32, 32)
    i[0] = i[0] // 2
  A[lane_id] = v[0]
```

シャッフルすると直接 `__shfl_xor_sync` に下げられます:

```c++
v_ptr[0] = v_ptr[0] + __shfl_xor_sync(0xFFFFFFFF, v_ptr[0], i_ptr[0], 32);
```

`T.ptx.*` / `T.cuda.*` には `cp_async` (LDGSTS)、 `cp_async.bulk.tensor` (TMA)、 `ldmatrix` / `stmatrix`、 `tcgen05.*` (ブラックウェル MMA)、 `atomic_add`、 `fence` なども含まれます 家族。 完全なリストについては、 `tvm.backend.cuda` バックエンド API リファレンスを参照してください。

## 同期された意味論

以下の 4 つの同期メカニズムは、GEMM および Flash Attention kernel で頻繁に登場します。 非同期エンジンや並列スレッドグループを制御し、使用ミスがサイレントの破損やデッドロックを引き起こすことがよくあります。

Mbarrier phase。Mbarrier は内部の位相ビットを使って異なる弾の到着を追跡します。 `T.ptx.mbarrier.try_wait(bar, phase)` barrier 内の位相が発信者の提示する `phase` ものと異なるまで待ち続けます。 barrier を繰り返し再利用する場合、各待機サイクル後に局所位相トラッカー(`phase ^= 1`)を反転させる必要があります。 そうでなければ、その後の待ち時間は即座に戻り、エンジンは書き込みされたメモリの一部だけを読み込むことがあります。 パート 3 の GEMM 章では、完全な位相追跡表が提供されています。

選挙。 `T.ptx.elect_sync()` warp 内のアクティブなレーンからレーン選択; 必ずしもレーン 0 を選ぶわけでも、すべての CTA がスレッドを選択するわけでもありません。 もし命令を発行できるスレッドが 1 本だけなら、warp レベルのガードも使用する必要があります。 第 3 部の GEMM kernel は `if warp_id == 0:` プラス `if T.ptx.elect_sync():` を使って `Tx.gemm_async` と `tcgen05.commit` を送信します。

warpgroup の barrier 名。 `T.cuda.cta_sync()` `__syncthreads()` に対応し、CTA 内のすべてのスレッドが到着する必要があります。 warpgroup が異なる役割のブランチに入ると、 `cta_sync()` を warpgroup のブランチのいずれかに配置できません。そうでなければ、他の warpgroup が到達できず、kernel はデッドロックされます。 ハードウェアは 16 個の名前付き barrier(ID 0 から 15)を提供します。 `T.cuda.warpgroup_sync(10)` warpgroup のスレッドを 1 つだけ同期させる。 異なる warpgroup は、同じハードウェア barrier を共有しないように `warpgroup_sync(wg_id + 10)` などの異なる ID を使用します。 パート 3、warp specialization 型 GEMM では、その全使用法が実演されます。

フェンス。Fence は、producer が consumer(通常は非同期エンジン)が読み込む前に次のように書き込むことを保証します:

| フェンス | 保証された注文 |
| --- | --- |
| `T.ptx.fence.proxy_async("shared::cta")` | Thread が共有メモリに書き込みた後、非同期プロキシ(TMA ストア/MMA)がそれを読み取ることができます |
| `T.ptx.fence.mbarrier_init()` | mbarrier を初期化した後、その後の到着や待機で使用可能です |
| `T.ptx.tcgen05.fence.after_thread_sync()` | `tcgen05` 保守党が writeback の端に柵を命じる; ステップ 8 と 9 で使用してください。TMA から MMA への経路は必須ではありません |

## インライン CUDA ソースコード

もし操作が本質的操作に対応していない場合、ソースコード文字列の `__device__` 関数を `T.cuda.func_call(name, *args, source_code=..., return_type=...)` に注入してコードを生成できます:

```python
SRC = r"""
__device__ __forceinline__ float my_relu(float x) {
  return x > 0.f ? x : 0.f;
}
"""

@T.prim_func
def k(A_ptr: T.handle, B_ptr: T.handle):
  A = T.match_buffer(A_ptr, (256,), "float32")
  B = T.match_buffer(B_ptr, (256,), "float32")
  T.device_entry(); bx = T.cta_id([1]); tx = T.thread_id([256])
  B[tx] = T.cuda.func_call(
    "my_relu",
    A[tx],
    source_code=SRC,
    return_type="float32",
  )
```

ソースコードはそのまま出力され、対応する呼び出し場所に接続されます。

```c++
__device__ __forceinline__ float my_relu(float x) {
  return x > 0.f ? x : 0.f;
}
// ...
B_ptr[tx] = my_relu(A_ptr[tx]);
```
