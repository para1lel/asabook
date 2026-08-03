---
title: "Buffer とメモリ"
createTime: 2026/08/01 00:00:00
permalink: /ja/gpupro/buffers-and-memory/
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

パラメータバッファは `T.match_buffer` でバウンドされ、kernel ボディ内の一時バッファは以下で紹介する 2 つの宣言 API を使って作成されます。 `A[i, j]` を使って要素にアクセスし、 `BufferRegion` を取得し `A[m0:m0+BM, 0:BK]` `A.ptr_to([i, j])` 要素ポインタを取得し `A.data` 元のデータポインタを取得します。

## バッファ宣言

バッファを作成するための基本的な API は以下の 2 つです。

- `T.alloc_buffer(shape, dtype, scope=..., ...)`: 新しいストレージを割り当て、 `AllocBuffer` ノードを生成し、対応する `Buffer` を返す。 `T.alloc_shared` と `T.alloc_local` はそれぞれ `scope="shared"` と `scope="local"` の略です。
- `T.decl_buffer(shape, dtype, data=..., ...)`: 既存のポインタ `data` 上で、ビューを宣言し、ストレージを割り当てしません。 プールや tensor メモリアドレスの一部にエイリアスを作成したり、既存のストレージを再解釈したりできます。 `data=None`、ストレージの割り当ては `alloc_buffer` と同じです。

バッファの `data` ポインタは不変です `Var`: `alloc_buffer` 作成 `decl_buffer` 受信。 すでにポインタ式を持っている場合は、まずその式を `Var` にバインドする必要があります。詳細は[データ型と式](/ja/gpupro/data-types-and-expressions/)を参照してください。

両 API は同じバッファディスクリプタを使用し、主なパラメータは以下の通りです:

| パラメータ | 意味 |
| --- | --- |
| `dtype` | `"float32"`、 `"float16"`、 `"float4_e2m1fn"` などの元素タイプ |
| `shape` | 論理形状は様々な次元拡張からなるタプルです |
| `layout` | 物理的マッピングについては[TIRx Layout API](/ja/gpupro/tirx-layout-api/)を参照してください。 `"default"` は密な行長音を示します |
| `elem_offset` / `allocated_addr` | `elem_offset` (または `byte_offset`) `data` 内で視界をオフセットに配置します。 `allocated_addr` tensor メモリ用に事前割り当てされたアドレスを格納します |
| `align` | データポインタのバイトアライメント |

`scope` パラメータ選択メモリ空間:

| 範囲 | 略語 | 記憶 |
| --- | --- | --- |
| `"global"` | デフォルト値 | デバイスグローバルメモリ |
| `"shared"` | `T.alloc_shared` | 静的共有メモリ(`__shared__`) |
| `"shared.dyn"` | プール | プールで管理される動的共有メモリ |
| `"local"` | `T.alloc_local` | スレッドごとのレジスタ |
| `"tmem"` | TMEM プール | ブラックウェル tensor メモリ |

```python
A = T.match_buffer(A_ptr, (M, K), "float16", align=16)   # parameter buffer
As = T.alloc_shared((BM, BK), "float16")                 # new shared tile
# Register accumulator.
acc = T.alloc_local((4,), "float32")
view = T.decl_buffer((BM, BK), "float16", data=As.data)  # a view over As
```

非 TMEM バッファの場合、ポインタベースのバッファは単にポインタにメタデータのセットを加えたものです。 要素にアクセスする際、コンパイラは layout に基づいてアドレスを計算します:

: :

addr(buffer[coord])=バッファ。 データ+elem_offset+layout。 apply (coord, shape=shape)["m"]

`layout.apply` 各物理軸のマッピングを返します。ここで `"m"` 成分は要素のオフセットです。 したがって、同じ論理アクセスでもバッファのメタデータ計算によって異なるアドレスが生成されます。 以下は `4×8` 領域で `B[i, j] = A[i, j] + 1` を実行しますが、 `B` を 4 つの方法で宣言しています。

```python
from tvm.tirx.layout import TileLayout, S

# Row-major.
B = T.match_buffer(p, (4, 8), "float32")
# Column-major.
B = T.match_buffer(
  p,
  (4, 8),
  "float32",
  layout=TileLayout(S[(4, 8):(1, 4)]),
)
# Shifted view.
B = T.match_buffer(
  p,
  (4, 8),
  "float32",
  elem_offset=64,
)
# Row stride 16.
B = T.match_buffer(
  p,
  (4, 8),
  "float32",
  layout=TileLayout(S[(4, 8):(16, 1)]),
)
```

生成される CUDA は `B[i, j]` に異なるインデックスを使用します。 ロード `A[i, j]` 常に `i*8 + j` を維持し、メタデータ `B` だけが変わります:

```c++
B_ptr[((i * 8) + j)]        = ...;   // row-major:        i*8 + j
B_ptr[((j * 4) + i)]        = ...;   // column-major:     j*4 + i
B_ptr[(((i * 8) + j) + 64)] = ...;   // elem_offset=64:   i*8 + j + 64
B_ptr[((i * 16) + j)]       = ...;   // row stride 16:    i*16 + j
```

## 共有メモリ

共有メモリは 2 種類に分かれます。コンパイル時に固定静的共有メモリ、起動時に決定される動的共有メモリ。 TIRx は動的共有メモリ管理のためのプールヘルパーも提供しています。

### 静的共有メモリ

`T.alloc_shared` (すなわち `scope="shared"`)は、コンパイル時にサイズが決定される最も単純な静的共有バッファを作ります。 まず、データを共有メモリに書き込み、 `cta_sync` を使ってブロック全体が書き込みを認識していることを確認し、その後結果を読み込みます。

```python
@T.prim_func
def smem_demo(A_ptr: T.handle, B_ptr: T.handle):
  A = T.match_buffer(A_ptr, (128,), "float32")
  B = T.match_buffer(B_ptr, (128,), "float32")
  T.device_entry()
  bx = T.cta_id([1])
  tx = T.thread_id([128])
  sm = T.alloc_shared((128,), "float32")   # static shared memory
  sm[tx] = A[tx]
  T.cuda.cta_sync()
  B[tx] = sm[tx] * T.float32(2.0)
```

標準的な `__shared__` 配列を生成します。 以下は無関係な CUDA コードの省略です:

```c++
extern "C" __global__ void __launch_bounds__(128)
smem_demo_kernel(float* __restrict__ A_ptr, float* __restrict__ B_ptr) {
  int tx = ((int)threadIdx.x);
  __shared__ alignas(64) float sm_ptr[128];      // T.alloc_shared
  sm_ptr[tx] = A_ptr[tx];
  __syncthreads();                               // T.cuda.cta_sync()
  B_ptr[tx] = sm_ptr[tx] * 2.0f;
}
```

### 動的共有メモリ

動的共有メモリ(`scope="shared.dyn"`)のサイズは、コンパイル時に書き込まれるのではなく、起動パラメータ `sharedMemBytes` によって指定されます。 各 kernel は動的共有割り当て 1 つしか持てず、それがアリーナです。 したがって、アリーナを一度割り当て、その後アリーナポインタの異なる `elem_offset` でビューを宣言する必要があります `T.decl_buffer`:

```python
# The single dynamic shared-memory arena.
arena = T.alloc_buffer((128,), "float32", scope="shared.dyn")
# View at offset 0.
As = T.decl_buffer(
  (64,),
  "float32",
  data=arena.data,
  scope="shared.dyn",
)
# View at offset 64.
Bs = T.decl_buffer(
  (64,),
  "float32",
  data=arena.data,
  elem_offset=64,
  scope="shared.dyn",
)
As[tx] = A[tx]
Bs[tx] = B[tx]
T.cuda.cta_sync()
C[tx] = As[tx] + Bs[tx]
```

同じ `extern __shared__` アリーナを共有している二つの視点があります。 読みやすくするために、アリーナは `smem` と名付けられ、無関係なコードは省略されています。

```c++
// The single dynamic-shared arena.
extern __shared__ __align__(64) float smem[];
smem[tx]      = A_ptr[tx];                       // As — view at offset 0
smem[tx + 64] = B_ptr[tx];                       // Bs — view at offset 64
__syncthreads();
C_ptr[tx] = smem[tx] + smem[tx + 64];
```

`alloc_buffer(scope="shared.dyn")` を 2 回呼び出すとエラーが発生します。なぜなら、動的共有メモリ割り当ては 1 つだけ許可されているからです。 静的共有メモリのサイズはコンパイル時に決定されます。例えば、 `__shared__ T x[N];`; 動的共有メモリは、ローンチ時にサイズが指定される単一のアリーナで、各バッファはアリーナ内の異なるオフセットのビューです。

::: note
TVM がどのように動的共有サイズを記録するか。Arena のサイズはコンパイル時に既知です。この例では `128` float があり、これは `512` バイトです。 降下時には、TVM はデバイス kernel の `tirx.kernel_launch_params` に `"tirx.use_dyn_shared_memory"` タグを追加します。 ホストランチャーは総バイト数を計算し、これを最後の起動引数として使用します:
```python
# device kernel attribute:
"tirx.kernel_launch_params": [
  "blockIdx.x",
  "threadIdx.x",
  "tirx.use_dyn_shared_memory",
]
# host-side launch call  (..., gridDim.x, blockDim.x, dyn_shared_bytes):
T.call_packed("dyn_kernel", A.data, B.data, C.data, 1, 64, 512)
```

実行時には、ここでの `512` が `cuLaunchKernelEx` 通話の `config.sharedMemBytes` となります。 ユーザーは手動で設定する必要はなく、 `shared.dyn` 割り当てのサイズから導き出されます。
:::

### スメンプール

`T.SMEMPool` 自動アリーナ管理。 バンプ割り当てを使ってオフセットを計算するため、各ビューを手動で `decl` する必要はありません。 `alloc` と `commit` に加え、バッファごとのご `align=`、MMA 互換のスウィズル layout 作成のための `alloc_mma` ヘルパー、そしてカーソルをロールバックしてスペースを再利用する `move_base_to` もサポートしています。

```python
pool = T.SMEMPool()                          # bump allocator over shared.dyn
As = pool.alloc((BM, BK), "float16", align=128)   # carve a tile
Bs = pool.alloc((BK, BN), "float16", align=128)
# MMA-compatible, with an inferred swizzle.
Cs = pool.alloc_mma((BM, BN), "float16")
pool.commit()                                 # finalize the pool's size
# pool.move_base_to(offset) rewinds the cursor to reuse space
```

下の TMEM プールは `SMEMPool` の上に建てられています。

## レジスタ

スレッドごとの一時データはレジスタ内に配置されます。配布には `T.alloc_local(shape, dtype)` (すなわち `scope="local"`)を使用します。 これは現在のスレッドにのみ属し、レジスタに格納されたローカル配列を生成します。

```python
r = T.alloc_local((4,), "float32")   # per-thread register array
for k in T.unroll(4):
  r[k] = A[tx, k]
# ... compute on r[0..3] ...
```

```c++
alignas(64) float r_ptr[4];          // per-thread, register-resident
r_ptr[0] = A_ptr[tx * 4 + 0];
r_ptr[1] = A_ptr[tx * 4 + 1];
// ...
```

::: note
バッファからのデフォルトのアライメント `alignas(64)`。 `data_alignment` デフォルトは `runtime::kAllocAlignment` で、64 バイトです。 CUDA コードジェンは、スレッドごとの配列 `local` を含め、各割り当てに追加します。 レジスタに保存された配列の場合、このアラインメントはパフォーマンスに影響を与えません。: インデックスがコンパイル時に解析できる限り、NVCC/ptxas はアグリゲートのスカラー置換(SROA)を用いてスレッドローカル配列をレジスタに昇格させます。 したがって、アドレス可能なローカルメモリは使用されません。 動的インデックスを使いローカルメモリにスポールする配列だけが過剰アライメントの影響を受けるのは珍しいことです。 レジスタローカルでの過剰アライメントは既知の問題です。将来的には、 `local` scope に dtype ナチュラルアラインメントを使うべきです。
:::

### スカラー

スカラーは本質的に、要素がつしかないレジスタ配列です。 サイズ 1 のバッファ `local` を直接割り当て、以下の方法でアクセスできます `[0]`:

```python
phase = T.alloc_local((1,), "int32")   # 1-element register array
phase[0] = 0
while phase[0] < 4:
  acc = acc + A[tx, phase[0]]
  phase[0] += 1
```

毎回 `phase[0]` を書き込むのは面倒なので、TIRx は同じ単一要素レジスタバッファを表すスカラー構文を提供し、名前で直接読み書きすることを可能にします。

```python
phase: T.int32 = 0                 # mutable scalar (sugar for the above)
while phase < 4:
  acc = acc + A[tx, phase]
  phase += 1

# Explicit form; assign by name (s = ..., not s[0]).
s = T.local_scalar("int32")
# A type-annotated assignment also creates a scalar.
acc: T.float32 = 0.0
```

解析後、両者とも同じ構造 TIRx が得られます。 パーサーは `phase: T.int32` を単一要素の `local` バッファに解析し、 `phase` と `phase += 1` を `phase[0]` と `phase[0] += 1` に分ける。 2 つの kernel に `tvm.ir.assert_structural_equal` を呼び出すとパスされます。 プリンターは明示的な `alloc_local` と `[0]` をスカラー構文として再出力することも可能です。 したがって、解析後、両者の間に違いはありません。両方とも `alignas(64) int phase_ptr[1];` に降格されます。 スカラーは単にこの `[0]` を省略します。 `T.local_scalar`、 `T.shared_scalar`、 `T.alloc_scalar` は明示的に scope を選択できます。

::: note
なぜ `Var` を使わないの?TIRx `Var` は不変の静的結合であり、以下の `T.let` と同じです。 スカラーは可変でなければならず、例えばループや accumulator で繰り返し割り当てられるため、繰り返し保存可能な単一要素バッファでサポートされ、 `Var` を使えません。
:::

### `let`

`T.let` は不変結合であり、これは `LetStmt` に対応します。 これはバッファではなく名前付き値を表し、導出定数を格納するのに適しています:

```python
n: T.let = M * K               # immutable binding (LetStmt)
half: T.let[T.int32] = N // 2  # ... with an explicit type
```

配列の代わりに標準的な C のスカラー変数を生成し、 `[0]` を必要としません。 例えば、実行時変数 `m` の `half: T.let = m * 2` は次のように生成されます。

```c++
int half = m * 2;     // the `let` -> a const-like local
```

値が変わらないため、簡略化器は自由に伝播や共通部分式の除去を行うことができます。 したがって、使用場所では `m * 2` を直接見たり、共有共通部分表現を一時的に見たりしても、必ずしもその `half` を保持するわけではありません。

::: note
なぜ不変のバインディングが必要なのか?値は変わらないため、算術アナライザは `LetStmt` を簡略化する際に `analyzer.Bind(var, value)` を呼び出し、定数境界、割り算や整列を表すモジュラー集合、範囲を含むすべての使用位置にこの値に関する結論を伝えることができます。 この情報はインデックスの簡素化、境界チェックの排除、アラインメントやベクトル化の決定に利用できます。 可変スカラーはメモリ負荷(`buf[0]`)であり、アナライザーはそれが変わらないと仮定できないため、これらの性質を伝播させることはできません。 `let` は依然として純粋な値であり、割り当てを必要としず、自由に線を組み込んだり、置換したり、CSE したりすることができます。 スカラーはロード/ストアの意味を持つ単一要素バッファです。
:::

## tensor メモリ

ブラックウェル tensor メモリ通常のスクラッチ scope のように直接割り当てることはできません。 kernel は明示的に warp 一様 `T.ptx.tcgen05.alloc` と `tcgen05.dealloc` イントリニシクスで要求・解放し、その後 tensor ビューを `T.decl_buffer(..., scope="tmem", allocated_addr=<column>, layout=<tmem layout>)` で宣言します。

`allocated_addr` は列オフセットを示し、これは必須のパラメータです。 tensor コアの dispatch が確認する。 したがって、集合 `allocated_addr` がなければ `T.alloc_buffer(scope="tmem")` は使用できません。 tensor メモリは直接アドレスを取ることができず、 `tcgen05` `mma`、 `ld`、 `st`、 `cp` を通じてのみアクセス可能です。

手動管理中、warp は割り当てアドレスを共有スロットに書き込み、異なる列オフセットの tensor ビューを `decl` 交換し、最後に warp がそれらを解放します。

```python
# Slot for the allocated base.
addr = T.alloc_shared((1,), "uint32")
# tcgen05.alloc is warp-uniform.
if warp_id == alloc_warp:
  T.ptx.tcgen05.alloc(T.address_of(addr), n_cols=512, cta_group=cta_group)
acc = T.decl_buffer((CTA_M, 512), "float32", scope="tmem",
  allocated_addr=0, layout=tmem_layout)   # view at column 0
# ... use acc as a gemm_async / copy_async operand ...
if warp_id == alloc_warp:
  T.ptx.tcgen05.relinquish_alloc_permit(cta_group=cta_group)
  T.ptx.tcgen05.dealloc(addr, n_cols=512, cta_group=cta_group)
```

この時点で、列のオフセットや `tmem_layout` (データパスの D/F layout)は手動で管理する必要があります。 下のプールは自動的に同じステップを生成します。

### TMEMPool(TMEMPool)

`T.TMEMPool` warp・ユニフォーム・アロック/デアロック、カラムバンプ割り当て、データパス layout をカプセル化する:

```python
# The pool is the kernel's SMEM pool.
tmem_addr = pool.alloc((1,), "uint32")
tmem_pool = T.TMEMPool(pool, total_cols=512, cta_group=cta_group,
  tmem_addr=tmem_addr)
acc = tmem_pool.alloc((CTA_M, 512), "float32")  # allocated_addr set for you
# Emit tcgen05.alloc from one warp.
tmem_pool.commit()
# ... use acc ...
# Emit tcgen05.dealloc from one warp.
tmem_pool.dealloc()
```

完全な例については、GEMM kernel のパート 3 を参照してください。

## バッファ API

`Buffer` はポインタ上のメタデータの集合です。上記の[バッファ宣言](#バッファ宣言) を参照してください。 したがって、ほとんどのメソッドはコンパイル時に形状、layout、インデックス演算、リターンポインタを変更し、実行時の操作自体は生成しません。 一般的な方法は以下の通りです:

| 方法 | 機能 |
| --- | --- |
| `B.data` | 元のデータポインタ(`Var`)、出力は `B_ptr` |
| `B.ptr_to([i, j])` | 要素を指す型付けポインタ(`address_of`)は出力 `&B_ptr[…]` |
| `B.vload([i], dtype="float32x4")` / `B.vstore([i], v)` | ベクトル化されたロード/ストア、出力 `*(float4*)(B_ptr + …)` |
| `B.view(*shape, layout=…)` | 同じストレージを新しい形状や layout で再解釈し、データをコピーせずに行えます |
| `B.local(*shape, layout=…)` | `local` バッファ内のプライベートレジスタスライスは現在のスレッドに属します |
| `B.permute(*dims)` | 軸を入れ替えた後の眺め、つまり転置 layout です |
| `B.access_ptr(mask, …)` | マスク付きアクセスポインター、または組み込みのもの `tvm_access_ptr`、領域を Intrinsic に渡すために使われます |

ポインター: `ptr_to` / `data`。 `ptr_to` 要素のアドレスをイントリシック関数またはインライン関数に渡します。 `data` は基本ポインタです:

```python
B[tx] = T.cuda.func_call(
  "ld",
  A.ptr_to([tx]),
  source_code=SRC,
  return_type="float32",
)
```

```c++
B_ptr[tx] = ld(&A_ptr[tx]);
// ptr_to([tx]) -> &A_ptr[tx]; A.data -> A_ptr
```

ベクター化されたアクセス: `vload` / `vstore`。複数の要素を同時に移動するためのワイド転送も参照してください[データ型と式](/ja/gpupro/data-types-and-expressions/):

```python
B.vstore([tx * 4], A.vload([tx * 4], dtype="float32x4"))
```

```c++
*(float4*)(B_ptr + tx * 4) = *(float4*)(A_ptr + tx * 4);
```

再構成/再解釈: `view` / `permute`。どちらもメタデータを変更するだけです。データポインタ自体は変わらず、変わるのはインデックス演算です。 `A.view(64, 4)` 256 要素を含むバッファを `64×4` として扱います。 `A.permute(1, 0)` 2 つの軸を交換する:

```python
A2 = A.view(64, 4)
y = A2[tx, 0] + A2[tx, 3]
# A2[tx, j] -> A_ptr[tx * 4 + j]
At = A.permute(1, 0)
z = At[i, j]
# At[i, j] -> A_ptr[j * 4 + i]
```

```c++
A2_ptr[tx * 4]  /* +3 */                 // view: row-major 64x4 index
At_ptr[(j * 4) + i]                       // permute: swapped strides
```

レジスタ: `local`。スレッド軸で `local` layout を分解し、現在のスレッドに属するフラットレジスタ束を返します。 tile プリミティブは頻繁にこれを用いています:

```python
R = T.alloc_buffer(
  (32, 8),
  "float32",
  scope="local",
  layout=TileLayout(S[(32, 8) : (1 @ laneid, 1)]),
)
Rl = R.local(8)          # this lane's 8 registers
```

```c++
alignas(64) float Rl_ptr[8];             // the lane's private registers
```
