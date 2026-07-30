---
title: 第 3 週
createTime: 2026/07/21 10:00:00
permalink: /ja/csdiy/cse291a-week3/
---

> あなたは私のすべての憧れ、私が敬い、愛するもののすべて。

第 3 週の内容は、CUDA のプログラミングモデルと行列積です。CUDA は「Compute Unified Device Architecture」の略ですが、文脈によって NVIDIA エコシステム内のさまざまなものを指します。デバイスアーキテクチャとしては、同一のハードウェアユニットを並べ、どのユニットでもさまざまな計算を実行できる GPU 設計を意味します。これらのユニットを==ストリーミングマルチプロセッサ==と呼びます。プログラミングモデルとしては、GPU がプログラマへ提供する抽象化、とりわけ==スレッドグループの階層構造==と==メモリ階層==を指します。ソフトウェアプラットフォームとしては CUDA プログラムを開発するためのソフトウェア群であり、cuDNN や cuBLAS のような計算ライブラリ、Triton や TileLang のような DSL もここに含められます。

![CUDA のプログラミングモデル](https://modal-cdn.com/gpu-glossary/terminal-cuda-programming-model.svg)

- **スレッド**は OS における概念と同じく、一連の逐次的な命令列を表し、CUDA の基本的なプログラミング単位でもあります。
- **スレッドブロック**は、shared memory を共有し、互いに同期できるスレッドの集まりです。一つのスレッドブロックは、まとめて一つの SM へスケジュールされます。
- **warp** は SM 内部の実行単位で、通常 32 スレッドからなります。warp 内のスレッドは同じ命令を同時に実行します。これを**単一命令・複数スレッド**といいます。分岐によって一部のスレッドだけが命令を実行するとき、ほかのスレッドは停止します。分岐を避け、同じ warp の全スレッドに同じ命令を実行させることが性能上重要です。分岐による性能低下を**warp divergence** と呼びます。
- **grid** は CUDA kernel の起動によって作られる複数のスレッドブロックを表します。各ブロックはほぼ独立で、任意の順序で各 SM へスケジュールできなければなりません。スレッドブロックの意味論は、計算タスクの並列度を表します。

## 最初の CUDA プログラム

filter 長 5 の Conv1d を実装します。メモリ律速の kernel を最適化する例です。

```python
def torch_conv1d(x, w):
  return F.conv1d(x.view(1, 1, -1), w.view(1, 1, -1)).view(-1)
```

基本実装では、各スレッドが連続する 8 個の出力を計算します。まず必要な $w$ の 5 要素と $x$ の 12 要素をレジスタへ読み込み、8 個の出力を計算して、そのまま $y$ へ書き戻します。

```cpp :collapsed-lines=10
__global__ void __launch_bounds__(THREADS, 1)
conv1d_k5_register_kernel(const __half* __restrict__ x,
                          const __half* __restrict__ w,
                          __half* __restrict__ y,
                          const long long n_out) {
  const float w0 = __half2float(__ldg(&w[0]));
  const float w1 = __half2float(__ldg(&w[1]));
  const float w2 = __half2float(__ldg(&w[2]));
  const float w3 = __half2float(__ldg(&w[3]));
  const float w4 = __half2float(__ldg(&w[4]));
  const long long gout = (long long)blockIdx.x * TILE
                         + (long long)threadIdx.x * VEC;
  if (gout >= n_out) return;

  float r[VEC + K - 1];
  if (gout + VEC <= n_out) {
    #pragma unroll
    for (int j = 0; j < VEC + K - 1; ++j) {
      r[j] = __half2float(__ldg(&x[gout + j]));
    }
  } else {
    const long long n_in = n_out + K - 1;
    #pragma unroll
    for (int j = 0; j < VEC + K - 1; ++j) {
      const long long idx = gout + j;
      r[j] = (idx < n_in) ? __half2float(__ldg(&x[idx])) : 0.f;
    }
  }

  #pragma unroll
  for (int j = 0; j < VEC; ++j) {
    r[j] = w0 * r[j]
             + w1 * r[j + 1]
             + w2 * r[j + 2]
             + w3 * r[j + 3]
             + w4 * r[j + 4];
  }

  if (gout + VEC <= n_out) {
    __half out[VEC];
    #pragma unroll
    for (int j = 0; j < VEC; ++j) out[j] = __float2half(r[j]);
    *reinterpret_cast<uint4*>(&y[gout]) =
      *reinterpret_cast<const uint4*>(out);
  } else {
    for (int j = 0; j < VEC && gout + j < n_out; ++j)
      y[gout + j] = __float2half(r[j]);
  }
}
```

- `__global__` は、CPU プロセスから起動される GPU kernel であることを表します。
- `__launch_bounds__` はブロック当たりのスレッド数と、各 SM で最低いくつのブロックを実行するかを指定します。コンパイラはこの指定に基づき、スレッドブロックが使う SM の資源（レジスタ、共有メモリ、warp など）を調整します。
- `__restrict__` は、そのポインタが対応するメモリ領域へアクセスする唯一の経路であることを示します。コンパイラはメモリの alias を排除でき、保守的なメモリロードや待機を避けて、グローバルメモリの読み書きをより積極的に最適化できます。
- `__ldg__` は、読み出し専用 cache を経由してグローバルメモリから明示的にロードします。読み出し速度やメモリ帯域幅を改善できる場合があります。  
  通常は `const T* __restrict__` と同じ効果になるので、ここでは実のところあまり役に立ちません。
- `reinterpret_cast<uint4*>` はベクトル化されたメモリアクセスに使い、1 命令で 16 byte を処理します。

$x$ への反復アクセスは L1 Cache によって処理されます。Nsight Compute で測定すると、DRAM 帯域幅は 80% に達しました。  
共有メモリ（手動で管理する L1 Cache）を使って、アクセスパターンをより規則的にすることもできます。共有メモリを使う際には bank conflict に注意が必要です。NVIDIA GPU は共有メモリを 32 個の bank に分け、アドレス $s$ を番号 $\lfloor s/4\rfloor\bmod 32$ の bank へ割り当てます。warp の一つの命令が一部の bank だけへアクセスを集中させ、ほかを空けてしまうと、帯域幅が無駄になります。  
ただし、ここでは DRAM 帯域幅がボトルネックなので、実際にはほとんど関係ありません。L1 Cache が少し遅くても大差はないのです。

```cpp :collapsed-lines=10
__global__ void __launch_bounds__(THREADS, 1)
conv1d_k5_kernel(const __half* __restrict__ x,
                 const __half* __restrict__ w,
                 __half* __restrict__ y,
                 const long long n_out) {
  __shared__ __align__(16) __half s[TILE + K - 1];
  const float w0 = __half2float(__ldg(&w[0]));
  const float w1 = __half2float(__ldg(&w[1]));
  const float w2 = __half2float(__ldg(&w[2]));
  const float w3 = __half2float(__ldg(&w[3]));
  const float w4 = __half2float(__ldg(&w[4]));
  const long long base = (long long)blockIdx.x * TILE;
  const int t = threadIdx.x;
  const long long n_in = n_out + K - 1;

  const long long g = base + (long long)t * VEC;
  if (g + VEC <= n_in) {
    *reinterpret_cast<uint4*>(&s[t * VEC]) =
      *reinterpret_cast<const uint4*>(&x[g]);
  } else {
    #pragma unroll
    for (int j = 0; j < VEC; ++j) {
      const long long idx = g + j;
      s[t * VEC + j] = (idx < n_in) ? x[idx] : __float2half(0.f);
    }
  }
  if (t < K - 1) {
    const long long idx = base + TILE + t;
    s[TILE + t] = (idx < n_in) ? x[idx] : __float2half(0.f);
  }
  __syncthreads();

  const int o = t * VEC;
  const long long gout = base + o;
  if (gout >= n_out) return;

  const uint4 v0 = *reinterpret_cast<const uint4*>(&s[o]);
  const uint2 v1 = *reinterpret_cast<const uint2*>(&s[o + VEC]);
  const __half* h0 = reinterpret_cast<const __half*>(&v0);
  const __half* h1 = reinterpret_cast<const __half*>(&v1);

  float r[VEC + K - 1];
  #pragma unroll
  for (int j = 0; j < VEC; ++j) r[j] = __half2float(h0[j]);
  #pragma unroll
  for (int j = 0; j < K - 1; ++j) r[VEC + j] = __half2float(h1[j]);

  #pragma unroll
  for (int j = 0; j < VEC; ++j) {
    r[j] = w0 * r[j]
             + w1 * r[j + 1]
             + w2 * r[j + 2]
             + w3 * r[j + 3]
             + w4 * r[j + 4];
  }

  if (gout + VEC <= n_out) {
    __half out[VEC];
    #pragma unroll
    for (int j = 0; j < VEC; ++j) out[j] = __float2half(r[j]);
    *reinterpret_cast<uint4*>(&y[gout]) =
      *reinterpret_cast<const uint4*>(out);
  } else {
    for (int j = 0; j < VEC && gout + j < n_out; ++j)
      y[gout + j] = __float2half(r[j]);
  }
}
```

同じく Nsight Compute で測定すると、DRAM 帯域幅は 90% で、ほぼ使い切れています。

![Nsight Compute のメモリワークロード分析](../../csdiy/image.png)

DRAM 以外の構成要素は、どれもほとんど遊んでいます。この種の kernel は単体でこれ以上どうにかしにくく、処理を一つ fuse するたびに DRAM の読み書きを一往復減らせます。Claude はまだ TileLang をうまく書けないようで、CUDA とまったく同じ形にして初めて同等の性能へ届きました。

```python :collapsed-lines=10
@tilelang.jit(target="cuda")
def _conv1d(N_out, K, block_N, threads):

  @T.prim_func
  def main(
    x: T.Tensor((N_out + K - 1,), "float16"),  # type: ignore
    w: T.Tensor((K,), "float16"),  # type: ignore
    y: T.Tensor((N_out,), "float16"),  # type: ignore
  ):
    VEC = block_N // threads  # halfs per thread (8 -> uint4 accesses)
    N_in = N_out + K - 1

    with T.Kernel(T.ceildiv(N_out, block_N), threads=threads) as bx:
      x_shared = T.alloc_shared((block_N + K - 1,), "float16")
      w_local = T.alloc_local((K,), "float32")
      r = T.alloc_local((VEC + K - 1,), "float32")
      out = T.alloc_local((VEC,), "float16")

      tx = T.get_thread_binding(0)
      base = bx * block_N + tx * VEC

      # Main tile (128-bit vectorized; scalar guarded in the last block)
      # plus K-1 halo elements.
      if base + VEC <= N_in:
        for j in T.vectorized(VEC):
          x_shared[tx * VEC + j] = x[base + j]
      else:
        for j in T.serial(VEC):
          x_shared[tx * VEC + j] = T.if_then_else(
            base + j < N_in, x[base + j], T.float16(0))
      if tx < K - 1:
        x_shared[block_N + tx] = T.if_then_else(
          bx * block_N + block_N + tx < N_in,
          x[bx * block_N + block_N + tx], T.float16(0))
      for k in T.serial(K):
        w_local[k] = T.cast(w[k], "float32")
      T.sync_threads()

      if base < N_out:
        for j in T.serial(VEC + K - 1):
          r[j] = T.cast(x_shared[tx * VEC + j], "float32")
        for j in T.serial(VEC):
          out[j] = T.cast(
            w_local[0] * r[j] + w_local[1] * r[j + 1]
            + w_local[2] * r[j + 2] + w_local[3] * r[j + 3]
            + w_local[4] * r[j + 4], "float16")

        if base + VEC <= N_out:
          for j in T.vectorized(VEC):
            y[base + j] = out[j]
        else:
          for j in T.serial(VEC):
            if base + j < N_out:
              y[base + j] = out[j]

  return main
```

IR が段階的に変換される様子と、コンパイル後の CUDA コードは[こちら](/webpage/conv1d-report.html)で確認できます。[公式ドキュメント](https://tilelang.com/tools/lower_trace.html)も参照してください。  
TileLang には本来、範囲外アクセスの検査が組み込まれていますが、`T.vectorized` を明示しなければメモリアクセスはベクトル化されません。
