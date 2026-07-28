---
title: 第 2 周
createTime: 2026/07/15 12:52:00
permalink: /csdiy/cse291a-week2/
---

> 不断反复的孤独夜晚, 心中祈愿的只是那微弱的温暖.  
> 犹如星星之间会相互吸引一般, 我梦想着, 与你相遇的瞬间.

第二周课接着上周讲计算图和自动微分, 然后开始讲计算优化的主题, 最后简单讲了一点矩阵乘法算子的分块优化. 自动微分的重点在于将反向传播的过程也建到计算图 (中间表示) 上, 将深度学习的语义与计算过程分离, 便于后续做系统级的优化.

这个课涉及的计算优化包括以下 4 个主题:

- 算子优化: 如何对各种计算需求和硬件写出高性能的算子 (在 GPU 上称为 kernel).
- 图级别优化: 例如算子融合、常量折叠、死代码消除等.
- 运行时环境: 计算调度与内存管理等.
- 多设备并行: 管理 10k+ 设备共同完成计算任务.

优化的重点在于降低内存读写的开销, 把处理器的算力打满.  
现代 ML 框架一般使用 ==步长格式== 表示张量: 每个维度 $i$ 记录 $\mathrm{shape}[i]$ 和 $\mathrm{stride}[i]$, 分别表示这一维的索引范围, 以及这一维索引增加 1 时在内存地址上增加的值. 这样可以把许多张量操作改为 shape 和 stride 的操作, 避免拷贝. 当然, 实际计算如果对内存访问有要求, 可以按需拷贝到连续位置.

$$A[i,j]=A.\mathrm{data}[\mathrm{offset}+i\cdot A.\mathrm{stride}[0]+j\cdot A.\mathrm{stride}[1]]$$

考虑一个简化的内存层级模型: 矩阵 A,B,C 放在 DRAM 上, 而计算在高速、容量小的寄存器上进行.

```python
def matmul(A: Float[Tensor, "n/v1 n v1", "DRAM"],
           B: Float[Tensor, "n/v2 n v2", "DRAM"],
           C: Float[Tensor, "n/v1 n/v2 v1 v2", "DRAM"]):
  for i in range(n/v1):
    for j in range(n/v2):
      c: Float[Tensor, "v1 v2", "reg"] = 0
      for k in range(n):
        a: Float[Tensor, "v1", "reg"] = A[i, k]
        b: Float[Tensor, "v2", "reg"] = B[j, k]
        c += outer(a, b)
      C[i, j] = c
```

A 和 B 的读取次数分别是 $n^3/v_2$ 和 $n^3/v_1$, C 的写入次数 $n^2$ 保持不变.  
减少读取次数是因为每次计算时 $v_2$ 个 B 元素共享了同一次 A 的读取, 从而将读取次数除以了 $v_2$.

<p style="text-align: center;">
  <img src="./hasumi.svg" alt="hasumi.svg" style="height: 300px; width: auto;">
</p>

这个方法称为 ==寄存器分块==. 对于更多的内存层级, 对应地可以做更多层的分块.

## PA1

PA1 是一个热身作业, 用 python 熟悉一下计算图定义和自动微分就好了.  
稍微难一点的是 Softmax 和 LayerNorm 的反向传播计算, 这里记录一下.

### Softmax

设 $y_i:=\mathrm{softmax}(x)_i=\mathrm e^{x_i}/\sum_j\mathrm e^{x_j}$, 已知 $g_i:=\partial\mathcal L/\partial y_i$, 要求 $\partial\mathcal L/\partial x_i$.

求导法则给出 $\partial y_j/\partial x_i=y_j(\mathbf 1_{\{i=j\}}-y_i)$, 所以
$$\frac{\partial\mathcal L}{\partial x_i}=\sum_{j=1}^N g_j\frac{\partial y_j}{\partial x_i}=y_i\left(g_i-\sum_{j=1}^N g_jy_j\right).$$

### LayerNorm

设 $\mu:=\frac 1N\sum_i x_i$, $\hat x_i:=x_i-\mu$, $\sigma^2:=\frac 1N\sum_i\hat x_i^2$, $s:=\sqrt{\sigma^2+\varepsilon}$, $y_i:=\hat x_i/s$,  
已知 $g_i:=\partial\mathcal L/\partial y_i$, 要求 $\partial\mathcal L/\partial x_i$.

这相当于一个“手动微分”:
$$
\begin{aligned}
\frac{\partial\sigma^2}{\partial x_i}&=\frac 2N\sum_{j=1}^N\hat x_j\frac{\partial\hat x_j}{\partial x_i}=\frac{2\hat x_i}N, \\
\frac{\partial s}{\partial x_i}&=\frac 1{2s}\frac{\partial\sigma^2}{\partial x_i}=\frac{\hat x_i}{Ns}, \\
\frac{\partial y_j}{\partial x_i}&=\frac 1s\frac{\partial\hat x_j}{\partial x_i}-\frac{\hat x_j}{s^2}\frac{\partial s}{\partial x_i}=\frac 1s\left(\mathbf 1_{\{i=j\}}-\frac 1N-\frac{y_iy_j}{N}\right), \\
\frac{\partial\mathcal L}{\partial x_i}&=\sum_{j=1}^N g_j\frac{\partial y_j}{\partial x_i}=\frac 1s(g_i-\mathrm{mean}(g)-y_i\cdot\mathrm{mean}(gy)).
\end{aligned}
$$