---
title: 第 3 周
createTime: 2026/07/21 10:00:00
permalink: /csdiy/cse291a-week3/
---

> 你是我所有的渴望, 我所敬仰和爱的一切.

第三周的内容是 CUDA 编程模型和矩阵乘法. CUDA 表示“统一的计算设备架构”, 视上下文不同可以代表 NVIDIA 生态里的各种东西. 作为设备架构, 代表各个硬件单元完全相同, 每个单元都能执行各种计算的这种 GPU 设计, 这些单元被称为 ==流式多处理器==. 作为编程模型, 代表 GPU 为程序员提供的抽象, 主要包括 ==线程组层次结构== 与 ==内存层次结构==. 作为软件平台, 是一系列用于开发 CUDA 程序的软件集合, cuDNN 和 cuBLAS 这类计算库, Triton 和 TileLang 这类 DSL 都可以归为这一类.

![cuda-programming-model.svg](https://modal-cdn.com/gpu-glossary/terminal-cuda-programming-model.svg)

- **线程**与操作系统中的概念类似, 描述一组串行的指令流, 也是 CUDA 的编程模式.
- **线程块**是一组共享 shared memory 且可同步的线程, 一个线程块会被统一调度到某个 SM 上.
- **线程束**是 SM 内部的执行单元, 通常是 32 个线程. 线程束内部的线程同时执行一条指令, 即**单指令多线程**. 若分支控制流导致仅有一部分线程需要执行该指令, 则其他线程会被暂停. 尽量避免分支, 保持同一线程束执行相同指令对性能很重要. 线程束分支导致的性能问题称为**线程束发散**.
- **网格**表示 CUDA Kernel 启动的多个线程块, 各个线程块应当是几乎独立的, 能以任意顺序调度到各个 SM 上. 线程块的语义表示计算任务的并行度.

## 第一个 CUDA 程序

我们考虑写一个 filter 长度为 5 的 Conv1d, 这是一个优化内存瓶颈 kernel 的例子.

```python
def torch_conv1d(x, w):
  return F.conv1d(x.view(1, 1, -1), w.view(1, 1, -1)).view(-1)
```

一个基本的实现是每个线程计算连续 8 个结果. 首先把用到的 5 个 $w$ 和 12 个 $x$ 读入寄存器, 然后计算这 8 个输出, 最后直接写回 $y$.

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

- `__global__` 表示这是一个由 CPU 进程启动的 GPU 核函数.
- `__launch_bounds__` 指定每个块的线程数, 以及每个 SM 最少运行的块数量. 编译器会根据这个配置线程块占用的 SM 资源 (寄存器、共享内存、线程束等).
- `__restrict__` 表示该指针是访问对应内存区域的唯一途径, 允许编译器消除内存别名, 避免保守的内存加载和等待, 更加激进地优化全局内存读写.
- `__ldg__` 表示强制通过只读缓存读取全局内存数据, 可能提升读取速度、优化内存带宽.  
  通常来说跟 `const T* __restrict__` 的效果是一样的, 其实没什么用.
- `reinterpret_cast<uint4*>` 用于向量化读写, 每条指令操作 16 个字节.

重复访问的 $x$ 会通过 L1 Cache 缓存, 用 Nsight Compute 测量显示 DRAM 带宽达到了 80%.  
也可以通过共享内存 (手动管理的 L1 Cache) 让访问模式更加规范. 使用共享内存需要注意访存冲突问题: NVIDIA GPU 将共享内存组织为 32 个 Bank, 地址 $s$ 对应编号 $\lfloor s/4\rfloor\bmod 32$ 的 Bank.  
线程束的同一条指令如果集中访问一部分 Bank, 另一部分空闲就会导致带宽浪费.  
当然在这里其实无关紧要, 瓶颈在 DRAM 带宽, L1 Cache 慢点也无所谓.

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

同样用 Nsight Compute 测量一下, DRAM 带宽 90%, 可以算是打满了.

![image.png](./image.png)

DRAM 之外的组件都闲得要死, 这种 kernel 就比较没救, 多一份 fuse 就少一份 DRAM 读写.  
Claude 好像还不太会写 TileLang, 只有跟 CUDA 写得一模一样才能达到同样的性能.

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

可以在[这里](/conv1d-report.html)查看 IR 的逐步变换和编译得到的 CUDA 代码, [官方文档](https://tilelang.com/tools/lower_trace.html).  
本来 TileLang 是自带越界检查的, 但是不显式写 `T.vectorized` 就没有向量化读写.
