---
title: 'Fast Transformer Decoding'
createTime: 2026/09/05 12:18:39
permalink: /papers/fast-transformer-decoding/
---

> [Noam Shazeer](https://www.noamshazeer.com/). 论文于 2019 年 11 月 6 日首次提交至 arXiv; 当前版本为 v1. [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150v1). [原始 PDF](/paper/fast-transformer-decoding.pdf). [DOI](https://doi.org/10.48550/arXiv.1911.02150). [TeX 源文件](https://export.arxiv.org/e-print/1911.02150v1). 精确的印刷版式和参考文献以原始 PDF 为准.

## 摘要

多头注意力层用于 Transformer 神经序列模型, 是在序列内部及序列之间传递信息的一种强大方案, 可替代 RNN. 由于可以沿序列长度并行计算, 这类层的训练通常快速而简单; 但增量推理无法采用这种并行方式, 往往较慢, 原因是反复加载大型 "键" 和 "值" 张量会产生内存带宽开销. 我们提出一种称为多查询注意力的变体, 让所有不同注意力 "头" 共享键和值, 从而大幅缩小这些张量, 降低增量解码所需的内存带宽. 实验表明, 由此得到的模型确实可以显著加快解码, 相比基线仅有轻微的质量下降.

<span id="section-1"></span>

## 1 引言

Transformer 神经序列模型 [Vas17] 已成为循环序列模型的一种常用替代方案. Transformer 依靠注意力层在序列内部及序列之间传递信息. Transformer 的一项主要难题是增量推理速度. 下文将说明, 在现代计算硬件上, Transformer 增量推理的速度受限于内存带宽: 系统必须重新加载编码注意力层状态的大型 "键" 和 "值" 张量. 接下来几节将回顾 Transformer 使用的多头注意力层, 分析其性能, 并提出一种架构变体 (多查询注意力), 以很小的质量损失大幅提升推理速度.

<span id="section-2"></span>

## 2 背景: 神经注意力

[Bah14] 提出的神经注意力是一种处理可变长度表示的有效工具. 神经注意力函数接收一个查询向量 $q$ 和 $m$ 个不同的 (键向量, 值向量) 对 (分别用矩阵 $K$ 和 $V$ 表示), 并生成输出向量 $y$. 输出 $y$ 是不同值向量的加权和, 权重由查询与各键的比较结果决定.

<span id="section-2-1"></span>

### 2.1 点积注意力

下面的代码给出一种常见形式, 其中的权重由查询与不同键的点积经过 softmax 后得到.

```python
def DotProductAttention(q, K, V):
  """对一个查询执行点积注意力.
  参数:
    q: 形状为 [k] 的向量
    K: 形状为 [m, k] 的矩阵
    V: 形状为 [m, v] 的矩阵
  返回:
    y: 形状为 [v] 的向量
  """
  logits = tf.einsum("k,mk->m", q, K)
  weights = tf.softmax(logits)
  return tf.einsum("m,mv->v", weights, V)
```

代码示例使用 TensorFlow 和 numpy 定义的 **einsum** 记法, 表示任意维张量之间的广义缩并. 在这种记法中, 一个方程会标出输入张量和输出张量的各个维度. 其计算在数值上等价于把每个输入广播到所有维度的并集, 逐分量相乘, 再对目标输出形状中没有出现的所有维度求和.

<span id="section-2-2"></span>

### 2.2 多头注意力

"Transformer" 序列到序列模型 [Vas17] 并行使用 $h$ 个不同的注意力层 (头), 作者称之为 "多头注意力". 这 $h$ 个不同层的查询向量来自输入向量 $x$ 经 $h$ 个不同的学习所得线性投影 $P_q$ 变换后的结果. 同样, 键和值来自由 $m$ 个不同输入向量组成的集合 $M$, 分别经过 $h$ 个不同的学习所得线性投影 $P_k, P_v$ 得到. 这 $h$ 个层的输出还会分别经过不同的学习所得线性投影 $P_o$, 然后相加. 为简化表述, 我们令输入向量和输出向量具有相同的维度 $d$. 计算过程可表示如下:

```python
def MultiheadAttention(
  x, M, P_q, P_k, P_v, P_o):
  """对一个查询执行多头注意力.
  参数:
    x: 形状为 [d] 的向量
    M: 形状为 [m, d] 的矩阵
    P_q: 形状为 [h, d, k] 的张量
    P_k: 形状为 [h, d, k] 的张量
    P_v: 形状为 [h, d, v] 的张量
    P_o: 形状为 [h, d, v] 的张量
  返回:
    y: 形状为 [d] 的向量
  """
  q = tf.einsum("d,hdk->hk", x, P_q)
  K = tf.einsum("md,hdk->hmk", M, P_k)
  V = tf.einsum("md,hdv->hmv", M, P_v)
  logits = tf.einsum("hk,hmk->hm", q, K)
  weights = tf.softmax(logits)
  o = tf.einsum("hm,hmv->hv", weights, V)
  y = tf.einsum("hv,hdv->d", o, P_o)
  return y
```

注意: [Vas17] 在 logits 上加入了一个常数缩放因子. 我们在代码中省略了它, 因为它可以并入线性投影 $P_q$ 或 $P_k$.

<span id="section-2-3"></span>

### 2.3 多头注意力 (批处理)

在实践中, 将多个查询合并成批处理会高效得多. 下面的代码增加了两种批处理. 首先, 我们从序列中的 $n$ 个不同位置生成查询. 这些查询与同一组键和值交互. 同时, 我们一次处理由 $b$ 个互不交互的序列组成的批次. 按照 [Vas17] 的做法, 在自回归模型中, 可以向 logits 加入一个 "掩码", 在不允许的位置填入 $-\infty$, 从而阻止信息向后流动.

```python
def MultiheadAttentionBatched(
  X, M, mask, P_q, P_k, P_v, P_o):
  """多头注意力.
  参数:
    X: 形状为 [b, n, d] 的张量
    M: 形状为 [b, m, d] 的张量
    mask: 形状为 [b, h, n, m] 的张量
    P_q: 形状为 [h, d, k] 的张量
    P_k: 形状为 [h, d, k] 的张量
    P_v: 形状为 [h, d, v] 的张量
    P_o: 形状为 [h, d, v] 的张量
  返回:
    Y: 形状为 [b, n, d] 的张量
  """
  Q = tf.einsum("bnd,hdk->bhnk", X, P_q)
  K = tf.einsum("bmd,hdk->bhmk", M, P_k)
  V = tf.einsum("bmd,hdv->bhmv", M, P_v)
  logits = tf.einsum("bhnk,bhmk->bhnm", Q, K)
  weights = tf.softmax(logits + mask)
  O = tf.einsum("bhnm,bhmv->bhnv", weights, V)
  Y = tf.einsum("bhnv,hdv->bnd", O, P_o)
  return Y
```

<span id="section-2-3-1"></span>

#### 2.3.1 批处理多头注意力的性能分析

为简化性能分析, 我们作出以下几个简化假设:

- $m=n$
- $k=v=\frac{d}{h}$, 与 [Vas17] 的建议相同
- $n\leq d$

算术运算总数为 $\Theta(bnd^2)$. 在这些简化假设下, 上述每个 `tf.einsum` 运算的复杂度都是 $O(bnd^2)$.

需要访问的内存总量等于所涉及张量大小的总和: $O(bnd+bhn^2+d^2)$. 第一项来自 $X$, $M$, $Q$, $K$, $V$, $O$ 和 $Y$, 第二项来自 logits 和 weights, 第三项来自投影张量 $P_q$, $P_k$, $P_v$ 和 $P_o$.

两者相除可得, 内存访问量与算术运算量之比为 $O(\frac{1}{k}+\frac{1}{bn})$. 现代 GPU/TPU 硬件的计算能力可能比内存带宽高两个数量级, 因此只有把这个比值保持在较低水平, 才能获得良好性能.

<span id="section-2-4"></span>

### 2.4 多头注意力 (增量式)

在某些场景中, 数据依赖关系使多个位置的查询无法并行处理. 自回归语言模型 (如 Transformer [Vas17]) 中的自注意力层就是一个例子. 每个位置产生的查询会关注从起始位置到该位置 (含该位置) 产生的所有键值对. 训练时, 真实目标序列已知, 因而可以采用与[第 2.3 节](#section-2-3)类似的高效并行实现. 但从训练好的模型生成内容时, 某一位置的自注意力层输出会影响下一位置生成的 token, 进而影响该层在下一位置的输入. 因此无法并行计算. 下面给出了增量计算这种自注意力层的代码.

```python
def MultiheadSelfAttentionIncremental(
  x, prev_K, prev_V, P_q, P_k, P_v, P_o):
  """多头自注意力 (单步).
  参数:
    x: 形状为 [b, d] 的张量
    prev_K: 形状为 [b, h, m, k] 的张量
    prev_V: 形状为 [b, h, m, v] 的张量
    P_q: 形状为 [h, d, k] 的张量
    P_k: 形状为 [h, d, k] 的张量
    P_v: 形状为 [h, d, v] 的张量
    P_o: 形状为 [h, d, v] 的张量
  返回:
    y: 形状为 [b, d] 的张量
    new_K: 形状为 [b, h, m+1, k] 的张量
    new_V: 形状为 [b, h, m+1, v] 的张量
  """
  q = tf.einsum("bd,hdk->bhk", x, P_q)
  new_K = tf.concat(
    [prev_K, tf.expand_dims(tf.einsum("bd,hdk->bhk", M, P_k),axis=2)],
    axis=2)
  new_V = tf.concat(
    [prev_V, tf.expand_dims(tf.einsum("bd,hdv->bhv", M, P_v), axis=2)],
    axis=2)
  logits = tf.einsum("bhk,bhmk->bhm", q, new_K)
  weights = tf.softmax(logits)
  o = tf.einsum("bhm,bhmv->bhv", weights, new_V)
  y = tf.einsum("bhv,hdv->bd", O, P_o)
  return y, new_K, new_V
```

<span id="section-2-4-1"></span>

#### 2.4.1 性能分析

我们采用与[第 2.3.1 节](#section-2-3-1)相同的简化假设.

在 $n$ 次调用中, 算术运算总数仍为 $\Theta(bnd^2)$.

在 $n$ 次调用中, 内存访问总量为 $\Theta(bn^2d+nd^2)$, 第一项来自 $K$ 和 $V$, 第二项来自 $P_q$, $P_k$, $P_v$ 和 $P_o$.

将内存访问量除以计算量可得, 内存访问量与算术运算量之比为 $\Theta(\frac{n}{d}+\frac{1}{b})$. 当 $n\approx d$ 或 $b\approx1$ 时, 该比值接近 1, 内存带宽因而会成为现代计算硬件上的主要性能瓶颈. 要让增量生成变得高效, 必须把这两项都降到 $\ll1$. $\frac{1}{b}$ 这一项较容易处理—在内存容量允许的情况下, 只需增大批次即可.

降低 $\frac{n}{d}$ 这一项更困难. 该项来自每一步重新加载表示记忆的 $K$ 和 $V$ 张量, 它们的大小为 $b h m k=bn^2$. 一种办法是限制序列长度 $n$. 另一种办法是减少被关注的位置数: 可以只关注局部邻域, 也可以像 [Liu18b], [Zha18b], [Pov18] 那样压缩记忆位置的数量. 本文提出一种与这些方法正交的方案来缩小 $K$ 和 $V$ 张量: 移除它们的 "头" 维度, 同时保留查询中的 "头" 维度.

<span id="section-3"></span>

## 3 多查询注意力

我们提出**多查询注意力**, 它是 [Vas17] 所述多头注意力的一种变体. 多头注意力由多个并行的注意力层 (头) 组成, 查询、键、值和输出分别采用不同的线性变换. 多查询注意力与其相同, 唯一的区别是不同的头共享一组键和值. 增量式多查询自注意力的代码与上文列出的多头注意力代码相同, 只需从 `tf.einsum` 方程中删除表示 $K$, $V$, $P_k$ 或 $P_v$ 的 "头" 维度的字母 "h".

```python
def MultiqueryAttentionBatched(
  X, M, mask, P_q, P_k, P_v, P_o):
  """多查询注意力.
  参数:
    X: 形状为 [b, n, d] 的张量
    M: 形状为 [b, m, d] 的张量
    mask: 形状为 [b, h, n, m] 的张量
    P_q: 形状为 [h, d, k] 的张量
    P_k: 形状为 [d, k] 的张量
    P_v: 形状为 d, v] 的张量
    P_o: 形状为 [h, d, v] 的张量
  返回:
    Y: 形状为 [b, n, d] 的张量
  """
  Q = tf.einsum("bnd,hdk->bhnk", X, P_q)
  K = tf.einsum("bmd,dk->bmk", M, P_k)
  V = tf.einsum("bmd,dv->bmv", M, P_v)
  logits = tf.einsum("bhnk,bmk->bhnm", Q, K)
  weights = tf.softmax(logits + mask)
  O = tf.einsum("bhnm,bmv->bhnv", weights, V)
  Y = tf.einsum("bhnv,hdv->bnd", O, P_o)
  return Y
```

```python
def MultiquerySelfAttentionIncremental(
  x, prev_K, prev_V, P_q, P_k, P_v, P_o):
  """多查询自注意力 (单步).
  参数:
    x: 形状为 [b, d] 的张量
    prev_K: 形状为 [b, m, k] 的张量
    prev_V: 形状为 [b, m, v] 的张量
    P_q: 形状为 [h, d, k] 的张量
    P_k: 形状为 [d, k] 的张量
    P_v: 形状为 [d, v] 的张量
    P_o: 形状为 [h, d, v] 的张量
  返回:
    y: 形状为 [b, d] 的张量
    new_K: 形状为 [b, m+1, k] 的张量
    new_V: 形状为 [b, m+1, v] 的张量
  """
  q = tf.einsum("bd,hdk->bhk", x, P_q)
  K = tf.concat(
    [prev_K, tf.expand_dims(tf.einsum("bd,dk->bk", M, P_k), axis=2)],
    axis=2)
  V = tf.concat(
    [prev_V, tf.expand_dims(tf.einsum("bd,dv->bv", M, P_v), axis=2)],
    axis=2)
  logits = tf.einsum("bhk,bmk->bhm", q, K)
  weights = tf.softmax(logits)
  o = tf.einsum("bhm,bmv->bhv", weights, V)
  y = tf.einsum("bhv,hdv->bd", O, P_o)
  return y, K, V
```

<span id="section-3-1"></span>

### 3.1 增量式多查询注意力的性能分析

我们采用与[第 2.3.1 节](#section-2-3-1)相同的简化假设.

在 $n$ 次调用中, 算术运算总数仍为 $\Theta(bnd^2)$.

在 $n$ 次调用中, 内存访问总量为 $\Theta(bnd+bn^2k+nd^2)$, 第一项来自 $x$, $q$, $o$ 和 $y$, 第二项来自 $K$ 和 $V$, 第三项来自 $P_q$, $P_k$, $P_v$, $P_o$.

将内存访问量除以计算量可得, 内存访问量与算术运算量之比为 $\Theta(\frac{1}{d}+\frac{n}{dh}+\frac{1}{b})$. 我们把不利的 $\frac{n}{d}$ 项缩小了 $h$ 倍. 理论上, 当批次大小 $b$ 较大时, 这应当能显著提升增量生成的性能. 实验部分将说明这种性能提升确实存在, 同时模型仍能保持较高质量.

<span id="section-4"></span>

## 4 实验与结果

<span id="section-4-1"></span>

### 4.1 实验设置

按照 [Vas17] 的做法, 我们在 WMT 2014 英德翻译任务上进行评估. 基线是一个 6 层编码器-解码器 Transformer 模型, 参数设置为 $d_{\mathrm{model}}=1024$ $d_{\mathrm{ff}}=4096$, $h=8$, $d_k=d_v=128$, 使用学习得到的位置嵌入, 并在 token 嵌入层与输出层之间共享权重. 基线模型和所有变体均有 2.11 亿个参数. 所有模型均训练 100,000 步 (~20 个 epoch). 每个训练批次包含 128 个样本, 每个样本由 256-token 的输入序列和 256-token 的目标序列组成 (为达到这一长度, 多个训练句子会拼接在一起). 模型在 32 核 TPUv3 集群上训练, 每个模型约需 2 小时. 我们使用 tensor2tensor 和 mesh-tensorflow 库中的实现. 所用配置可在 [to be added before publication] 中找到, 其中包括学习率、dropout、标签平滑等细节.

在 "multi-query" 模型中, 我们把模型中的所有注意力层替换为多查询注意力. 其中包括编码器自注意力、解码器自注意力和编码器-解码器注意力层. 我们把前馈隐藏层的宽度从 4096 增加到 5440, 使总参数量与基线相同.

为了说明局部注意力与多查询注意力彼此正交, 我们还训练了基线模型和多查询模型的 "local" 版本, 其中解码器自注意力层 (其他注意力层不变) 只关注当前位置及前 31 个位置.

缩小 $K$ 和 $V$ 的另一种简单办法是减少头数 $h$, 和/或降低键和值的维度 $k$ 和 $v$. 我们训练了若干此类模型用于比较, 同样加宽前馈隐藏层, 使总参数量与基线相同.

我们还在 Billion-Word Language Modeling Benchmark [Che13] 上, 使用 "transformer-decoder" 语言模型进行了一组类似的实验. 基线采用 6 层模型, 参数设置为 $d_{\mathrm{model}}=1024$ $d_{\mathrm{ff}}=8192$, $h=8$, $d_k=d_v=128$. 基线和所有变体的总参数量均为 1.92 亿. 我们使用 64K-token 的批次训练 136K 步 (10 个 epoch). 每个模型同样在 32 核 TPUv3 集群上训练约 3 小时.

<span id="section-4-2"></span>

### 4.2 模型质量

[表 1](#table-01) 给出了机器翻译实验的结果. 我们使用贪心最大似然解码处理开发集, 并通过 sacrebleu `"sacrebleu -t wmt13 -l en-de -tok intl"` 计算 BLEU 分数. 表中还列出了开发集上的每子词 token 困惑度. 按这两项指标衡量, 多查询注意力模型似乎略逊于基线, 但远比降低 $h$, $d_k$ 和 $d_v$ 的其他方案接近基线.

我们还使用贪心解码和束搜索 (beam 4, $\alpha=0.6$) 解码测试集, 并通过 sacrebleu `"sacrebleu -t wmt14 -l en-de -tok intl"` 进行评估, 以验证结果. 多查询模型的表现仍与基线相近, 而且在 beam-4 解码时取得了最高的 BLEU 分数 (28.5).

[表 3](#table-03) 给出了十亿词语言建模基准的结果. 模型按开发集上的每词 (而非每子词 token) 困惑度评估. 结果与翻译实验相似. 多查询注意力模型略逊于基线, 但明显优于降低 $h$, $d_k$ 和 $d_v$ 的其他方案.

<span id="section-4-3"></span>

### 4.3 速度

[表 2](#table-02) 给出了各模型的训练和推理时间. 训练和推理速度均在一台 TPUv2 (8 核) 上评估. 基础模型的一个训练步骤 (包含上述 32,768 个输入 token 和 32,768 个目标 token) 耗时 433ms, 多查询模型耗时 425ms. 除以 32,768 后可得, 每个 (输入 token + 目标 token) 的训练时间为 $13.2\mu\mathrm{s}$, 如[表 2](#table-02) 所示.

我们在包含 1024 个序列 (每核 128 个) 的批次上运行增量式贪心推理, 源序列和目标序列的长度均为 128 个 token. [+1] 对基线模型而言, 编码器部分耗时 222ms, 解码器的每个增量步骤耗时 47ms. 分别除以相应的 token 数可得, 编码器的摊销推理时间为每 token $1.7\mu\mathrm{s}$, 解码器则要大得多, 为每 token $46\mu\mathrm{s}$, 如[表 2](#table-02) 所示. 对多查询模型而言, 编码器耗时 195ms, 解码器每步耗时 3.9ms, 摊销到每个 token 后分别为 $1.5\mu\mathrm{s}$ 和 $3.8\mu\mathrm{s}$. [表 2](#table-02) 同时给出了这些数值和束搜索的相应结果.

[+1]: 由于系统限制要求使用固定形状, 我们在解码器自注意力实现中采用了填充和掩码. 因此, 记忆张量会填充到最大长度 (128); 使用局部注意力时则填充到窗口大小 (32). 每个解码步骤的耗时因而相同. 另一种实现可以让张量随增量逐步增长, 从而在序列开始附近节省时间.

<span id="table-01"></span>

![WMT14 英德翻译结果](./fast-transformer-decoding/table-01.png)

**表 1.** WMT14 英德翻译结果.

<span id="table-02"></span>

![WMT14 英德翻译的摊销训练与推理成本](./fast-transformer-decoding/table-02.png)

**表 2.** 序列长度为 128 时 WMT14 英德翻译任务的摊销训练与推理成本. 所列数值的单位为每个输出 token 所需的 TPUv2 微秒数.

<span id="table-03"></span>

![十亿词语言建模基准结果](./fast-transformer-decoding/table-03.png)

**表 3.** Billion-Word LM 基准结果.

<span id="section-5"></span>

## 5 结论

我们提出了多查询注意力, 它是多头注意力的一种替代方案, 在增量场景中的内存带宽需求低得多. 我们认为, 这能让基于注意力的序列模型更广泛地用于对推理性能要求较高的应用.
