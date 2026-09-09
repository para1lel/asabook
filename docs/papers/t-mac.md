---
title: 'T-MAC: CPU Renaissance via Table Lookup'
createTime: 2026/09/09 16:24:37
permalink: /papers/t-mac/
pageClass: paper-reading
---

> [Jianyu Wei](https://kaleid-liner.github.io/about/more/) [+internship], [Shijie Cao](https://caoshijie0501.github.io/) [+corresponding], [Ting Cao](https://www.microsoft.com/en-us/research/people/ticao/) [+corresponding], [Lingxiao Ma](https://xysmlx.github.io/), [Lei Wang](https://dblp.org/pid/181/2817-222) [+internship], [Yanyong Zhang](http://staff.ustc.edu.cn/~yanyongz/) 和 [Mao Yang](https://www.microsoft.com/en-us/research/people/maoyang/). 论文于 2024 年 6 月 25 日首次提交至 arXiv; 当前版本为 v2, 提交于 2025 年 3 月 25 日; 发表于 [EuroSys 2025](https://doi.org/10.1145/3689031.3696099). [T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge](https://arxiv.org/abs/2407.00088v2). <a href="/paper/t-mac.pdf" target="_blank" rel="noopener noreferrer">原始 PDF</a>. [DOI](https://doi.org/10.1145/3689031.3696099). [TeX 源码](https://export.arxiv.org/e-print/2407.00088v2). 精确的印刷排版与参考文献以原始 PDF 为准.

[+internship]: 此项工作在微软研究院实习期间完成.

[+corresponding]: 通讯作者.

## 摘要

在边缘设备上部署大语言模型 (LLM) 对提升端侧智能越来越重要. 权重量化可以减小 LLM 在设备上的内存占用. 然而, 低比特 LLM 在推理时需要执行低精度权重与高精度激活之间的**混合精度矩阵乘法 (mpGEMM)**. 现有系统并不原生支持 mpGEMM, 只能先反量化权重, 再进行高精度计算. 这种间接方式会带来显著的推理开销.

本文提出 T-MAC, 一种基于查找表 (LUT) 的方法, 用于在 CPU 上高效推理低比特 LLM (即权重量化 LLM). T-MAC 无需反量化即可直接支持 mpGEMM, 同时消除乘法并减少所需的加法. 具体而言, T-MAC 把传统的以数据类型为中心的乘法转换为逐比特查表, 从而提供统一且可扩展的 mpGEMM 方案.

我们的 LUT 内核随权重比特宽度线性扩展. 在低比特 Llama 和 BitNet 模型上的评估表明, 与 *llama.cpp* 相比, T-MAC 的吞吐量最高提升 $4\times$, 能耗最多降低 70%. 对于 BitNet-b1.58-3B, T-MAC 在 M2-Ultra 上使用单核时可达到 30 tokens/s 的生成吞吐量, 使用 8 核时达到 71 tokens/s; 在 Raspberry Pi 5 上也能达到 11 tokens/s. T-MAC 采用基于 LUT 的计算范式, 使低比特 LLM 可以在不牺牲计算效率的前提下实际部署到资源受限的边缘设备上. 系统已在 [https://github.com/microsoft/T-MAC](https://github.com/microsoft/T-MAC) 开源.

<span id="section-1"></span>

## 1 引言

越来越多的大语言模型 (LLM) 被部署到智能手机, 桌面计算机和机器人等客户端设备上, 用于提供前所未有的智能服务, 实时任务响应和用户数据保护. 典型例子包括部署在 iPhone 上的 Phi-3-mini-4bit [Abd24], 部署在 Pixel 5 上的 Llama-2-7B-4bit 和部署在 Apple M2 Ultra 上的 Llama-2-13B-4bit [Lla23a], 以及近期推出的 Microsoft Copilot+PC [Cop24], 它让端侧 LLM 与云端 LLM 协同运行.

由于硬件资源有限, 低比特权重量化是端侧 LLM 推理不可缺少的技术. LLM 的推理质量对精度损失也有较强的鲁棒性. 除 4-bit 外, 3-bit, 2-bit 乃至 1-bit 模型也在不断出现 [Du24, Xu24h, Wan23, Che24b]. 相比之下, 激活中存在离群值, 因此激活量化无法沿着同样的趋势降低精度. 计算的两个操作数因而具有不对称的精度和比特宽度, 例如 W4A16, W2A16 或 W1A8 [+1].

另一方面, 当前的通用硬件仍只支持固定比特宽度和对称操作数, 无法支持这些多样化的混合精度. 即使部分研究工作支持比特宽度不对称的操作数, 其比特宽度依然固定, 例如 W4A8 [Gop20]. 因此, 计算内核必须转换或反量化低比特权重, 使其与激活精度一致, 再交由硬件计算.

这种转换会引出两个明显的问题. *(i)* 从性能看, 转换开销抵消了降低比特数带来的收益. 我们的评估 (参见 [图 6](#figure-06)) 表明, 在大多数情况下, 将比特数从 4 bit 降至 1 bit 反而会增加延迟. *(ii)* 从开发看, 每一种混合精度都要单独设计数据布局和内核. 例如, W3 与 W2 的数据布局以及交错或 swizzling 方法完全不同. 内核也必须重新设计, 才能匹配相应布局. 因此, 要在设备上部署 LLM, 一个根本问题是如何直接且高效地支持低比特权重与高比特激活之间的 mpGEMM (混合精度通用矩阵乘法).

本文的目标是设计一种不依赖硬件数据类型和量化算法比特宽度的 mpGEMM 内核, 使性能随着比特数下降而获得可扩展的提升. 为实现这一目标, 我们的*核心思路*不是沿用主流的以数据类型为中心的计算方式, 而是利用标准乘法算法中的*逐比特*计算. 也就是说, 两个数相乘可以改写为: 让其中一个数分别乘以另一个数的每一位, 再移位并累加各个部分积. 激活矩阵与权重矩阵之间的 mpGEMM 可分解为若干个激活矩阵与 1-bit 矩阵之间的 mpGEMM (数量等于权重的比特宽度), 最后累加部分结果. 因而, 这种方法可以支持激活与权重的任意比特宽度组合.

查表是实现逐比特计算的一种可行方法 [Par23b]. 由于 1 bit 只能表示两个值, 例如 1/-1, 1-bit 向量的比特模式数量有限. 例如, 若把 1-bit 矩阵划分为由 4 个元素组成的向量组, 每组可能出现的比特模式 (如 [1,1,1,-1] 和 [1,1,-1,-1]) 只有 $2^{4}$ 种. 给定一个激活后, 可以先计算它与所有可能比特模式的结果并存入表中. 随后, 激活与 1-bit 矩阵之间的 mpGEMM 就转换为以权重中的各个比特模式为索引查表, 再通过加法累积查得的结果. 这样一来, mpGEMM 被简化为查表与加法, 不再需要乘法.

基于 LUT 的逐比特 mpGEMM 虽然能减少乘法, 但当前硬件已针对乘法做了高度优化, 因而很难在真实设备上高效实现. 逐比特 LUT 与传统 mpGEMM 的*显著差异*在于, LUT 方法的两个操作数是表和索引矩阵, 而不是激活和权重. 这两个操作数的数据格式与布局因此直接关系到推理速度. *(i)* 第一个难点是, 激活和权重通常采用连续访问, 而表访问是随机的. 要获得最终的推理性能, 表能否驻留在高速片上内存中尤其重要. *(ii)* 但片上内存容量有限, 且 LUT 方法比传统 mpGEMM 占用更多片上内存. 原因是 LUT 必须保存激活向量与所有可能比特模式相乘的结果. 这一数量相对于激活本身呈指数增长.

针对这些问题, 本文提出 mpGEMM 内核库 T-MAC. 如 [图 1](#figure-01) 所示, 它以逐比特计算和 LUT 实现为基础, 为激活与权重的任意混合比特宽度提供统一且可扩展的方案. 为减轻 LUT 随机访问的开销, 我们从系统和算法两方面提出技术, 使 LUT 能驻留于最快的片上内存并支持并行查找. 在系统层面, 我们提出*以 LUT 为中心的数据布局*: 通过轴重排和分块, 使 LUT 驻留在寄存器等片上内存中, 充分复用每张表, 同时减少与 LUT 争用片上内存的临时结果. 在算法层面, 我们提出*表量化*与*镜像合并*, 用来缩小表的规模.

我们在边缘设备上普遍配备的 CPU 处理器上实现了 T-MAC, 甚至包括 Raspberry Pi. 我们发现, T-MAC 能使 CPU 上的 LLM 推理速度达到甚至超过同一设备上的 GPU, 主要原因是 T-MAC 没有转换开销, 且查表减少了总操作量. *因此, 本文首次给出了利用广泛可用的 CPU, 而不依赖 GPU 在边缘设备上部署 LLM 的实用方案.* 我们在 Apple M2 Ultra, Jetson AGX Orin, Surface Book 3 和 Raspberry Pi 5 等典型边缘设备上评估了 T-MAC. 与 CPU 上当前最先进的 llama.cpp [Lla23a] 相比, T-MAC 内核最高可加速 $6.6\times$, 平均加速 $3.6\times$. 对 Llama-2-7B-2bit 模型 [Lla23b], 端到端 LLM 推理可加速 $2.8\times$. 即使在 Raspberry Pi 上运行 BitNet-b1.58-3B 模型 [Wan23], 推理速度也能达到 11.1 tokens/s. T-MAC 的能效也很高, 与 llama.cpp 相比可节省 60-70% 的能耗.

我们的贡献可概括如下:

- T-MAC 把以数据类型为中心的乘法转换为*逐比特*查表, 得到统一且可扩展的 mpGEMM 设计.
- 我们从系统和算法两方面提出技术, 让表驻留在最快的内存中并支持并行查找.
- 我们实现了 T-MAC 内核库和端到端推理系统, 显著加快 LLM 推理并降低能耗.

<span id="figure-01"></span>

![T-MAC 与 mpGEMM 通用做法的对比.](./t-mac/figure-01.png)

**图 1.** T-MAC 与 mpGEMM 通用做法的对比.

<span id="section-2"></span>

## 2 背景与动机

<span id="section-2-1"></span>

### 2.1 边缘设备上的 LLM

大语言模型 (LLM) 的出现改变了自然语言处理领域, 为人机交互和个性化助手开辟了新的空间. 将 LLM 直接部署到智能手机, 桌面计算机和机器人等边缘设备上, 已成为计算领域的重要前沿方向, 有望让这些设备具备前所未有的智能和自主能力.

**边缘 LLM 的优势.** 在边缘设备上部署 LLM 有多项明显优势. 端侧处理能大幅缩短响应延迟, 这对自动驾驶车辆和交互式机器人等时延敏感的应用十分重要. 本地处理数据还会把敏感信息留在设备内, 从而加强用户隐私并降低数据泄露风险. 另一项重要优势是不依赖网络所带来的运行可靠性: 无论网络是否可用, 是否稳定, 设备都能持续提供功能.

**边缘 LLM 的挑战.** 在边缘设备上部署 LLM, 首要难题是容纳这些模型需要大量内存. LLM 往往含有数十亿个参数. 例如, FP16 精度的 LLAMA-2-7B 至少需要 14GB 内存才能装入模型. 而边缘设备的内存资源通常有限, 这对端侧部署构成了严峻限制.

除了内存容量, LLM 对计算能力和内存带宽的需求也是边缘部署的一大障碍. 为满足单个用户的实时交互, 边缘设备常以单实例方式处理数据, batch size 通常为 1. LLM 推理可分为 prefill 和 decode 两个阶段. Prefill 阶段会对所有输入 token 应用自注意力机制, 其中包含计算密集的矩阵-矩阵乘法. 然而, 一旦生成 Key-Value (KV) cache, decode 阶段就会成为瓶颈. 在这一阶段, 每生成一个后续 token, 都需要加载并处理整个模型, 对应的是内存密集的矩阵-向量乘法.

第三项挑战是功率或能效, 对智能手机和机器人等电池供电的边缘设备尤其重要. 这类设备需要依靠有限的能量储备长时间运行, 因此必须考虑能效.

<span id="section-2-2"></span>

### 2.2 低比特 (权重量化) LLM

LLM 推理十分依赖内存, 因而需要在不显著损害模型性能的前提下缩小模型的内存占用. 权重量化是实现这一平衡的一项主要技术.

权重量化会降低模型参数的精度, 从而让模型占用更少内存, 并可能利用低精度算术加快计算 [Det22c, Fra22, Lin23d]. 如今, 越来越多的 LLM 会发布专门面向边缘环境或其他资源受限环境的 4-bit 版本 [Gem23, You24a]. 近期研究进一步探索了在 LLM 中采用 2-bit 和 1-bit 权重表示的可行性 [Du24, Xu24h, Wan23]. 从根本上说, 权重量化所选的比特宽度或精度体现了计算效率与模型准确率之间的权衡.

<span id="section-2-3"></span>

### 2.3 低比特 LLM 的部署挑战

边缘部署已经离不开低比特 LLM. 事实上, 许多边缘 LLM 系统和实现都在积极采用低比特技术 [Lla23a, Int18]. 但部署低比特 LLM 会带来特有的计算挑战, 尤其是如何容纳大多数硬件架构并不原生支持的混合精度运算, 以及如何管理不同部署场景所需的多种比特宽度与精度级别. 只有解决这些问题, 低比特 LLM 才能在边缘计算中充分发挥潜力并实现无缝集成.

**混合精度 GEMM/GEMV** 使用低比特 (权重量化) LLM 会形成一种低精度权重与相对高精度激活相结合的计算范式. 这要求标准矩阵乘法运算 (即 GEMM, GEMV) 转向混合精度运算 (即 mpGEMM, mpGEMV). 然而, CPU, GPU 和 NPU 等当前硬件架构都不原生支持混合精度运算. 这些架构传统上针对两个操作数采用相同数据类型和精度级别的标准运算做了优化.

针对这一限制, 现有系统采用基于反量化的间接方法: 将低精度权重转换回较高精度, 使其与激活精度一致. 这样便能用高精度 GEMM 完成低比特 LLM 推理. 例如, *Intel Neural Compressor* 和 *llama.cpp* 所用的系统都依赖这种反量化技术. 但这类方法有效的前提是反量化不会成为瓶颈, 并且能与内存加载重叠. 这种间接方法最终仍会退回高精度计算, 因而无法充分利用低比特权重减少内存用量, 可能加快计算等优势.

**比特宽度/精度的多样性** 除了混合精度运算本身的难题, 不同部署场景所需的比特宽度和精度各不相同, 又增加了复杂度. 根据任务难度和部署环境的具体要求, 系统可能选择多种比特宽度来优化性能. 不存在一种比特宽度或精度设置可以普遍满足所有用例的不同需求. 因此, 计算方法必须支持一系列低比特宽度, 才能适应边缘计算任务的广泛需求.

<span id="section-2-4"></span>

### 2.4 面向量化模型的 LUT 计算

量化模型计算的一个新趋势是采用基于查找表 (LUT) 的方法. 对于权重和激活都被量化为 4-bit, 2-bit 或 1-bit 等精度的量化卷积神经网络 (CNN), DeepGEMM [Gan23a] 会预先计算权重与激活所有可能的乘积并存入查找表, 在推理时高效访问这些结果, 以免执行代价高昂的乘加运算. 向量量化是另一个例子: 当激活采用向量量化时, MADDNESS [Bla21a] 和 LUT-NN [Tan23a] 同样把 GEMM 计算转换为查表.

在低比特 LLM, 即仅权重量化的 LLM 中, 研究人员已在 GPU 上探索了 LUT 方法 [Par23b, Mal23]. 这些方法利用 GPU 的 shared memory 或 cache 存储和访问查找表. 然而, 尽管理论上的计算复杂度有所降低, 其实际内核性能仍不及 [Nvi24a, Bit24a] 中基于反量化的内核. 例如, 在 A100 GPU 上使用真实 Llama-2 模型的权重矩阵形状测试时, 对于 $W_{\text{INT4}}A_{\text{FP16}}$, $W_{\text{INT2}}A_{\text{FP16}}$ 和 $W_{\text{INT1}}A_{\text{FP16}}$ mpGEMV, LUT-GEMM 内核 [Par23b] 的平均延迟分别比 BitBLAS [Bit24a] 的反量化内核高 $2.34\times$, $1.87\times$ 和 $1.75\times$. 这种不理想的内核性能源于 GPU 固定架构的限制: 查找表的存储容量不足, 或者表访问速度不够快. 相比之下, 基于 LUT 的混合精度 GEMM/GEMV 在 CPU 上仍无人探索. 本文率先研究这一方向, 考察把 LUT 方法用于 CPU 低比特 LLM 推理的可行性及其性能影响.

<span id="section-3"></span>

## 3 设计

<span id="figure-02"></span>

![图 2. T-MAC 设计概览.](./t-mac/figure-02.png)

**图 2.** T-MAC 设计概览.

现有混合精度 GEMM 实现通常需要逐案设计. 激活与权重的每种位宽组合, 如 W4A16 和 W2A8, 都需要特定的权重布局与计算内核. 例如, W3 布局可以把其中 2 bit 与另外 1 bit 分开打包, 再用不同的交错或 swizzling 方法实现内存对齐或快速解码. 对应的计算内核随后还要把这种特定布局解包为硬件支持的数据类型才能执行.

为给出统一且可扩展的混合精度 GEMM 方案, 本文依据 [公式 1](#equation-01) 的线性等价变换, 将主流的*以数据类型为中心*的计算改为*逐比特*计算. 在混合精度 GEMM 中, $A$ 和 $W$ 分别为激活矩阵与权重矩阵, $n$ 是权重位宽, $W_i$ 是 $W$ 的各个位平面矩阵.

<span id="equation-01"></span>

$$
A\times W=A\times(\sum^{n-1}_{i=0}2^{i}W_{i})=\sum^{n-1}_{i=0}2^{i}A\times W_{i}
$$

这样一来, 多种权重布局可归一为统一的 1-bit 矩阵布局, 多种计算内核也可归一为激活矩阵与 1-bit 矩阵之间的乘法. 此外, 逐比特计算使计算成本能够随位宽降低而线性下降.

本文用 LUT 方法实现这种逐比特布局与乘法 ([第 3.1 节](#section-3-1)), 并提出以 LUT 为中心的数据布局 ([第 3.2 节](#section-3-2)) 与表压缩方法 ([第 3.3 节](#section-3-3)), 使查找表能够驻留在寄存器中, 并实现最快的并行查表.

<span id="section-3-1"></span>

### 3.1 T-MAC 算法

[图 2](#figure-02) 和算法 1 展示了 T-MAC 的设计. 在离线准备阶段 (第 29 至 35 行), 一个 $n$-bit 权重矩阵会被拆成 $n$ 个 1-bit 矩阵. 由于 1 bit 只能表示两个值, 一组 $g$ bit 仅有 $2^g$ 种排列. 在线阶段可用每个形状为 $[1,g]$ 的激活分组预先计算这些排列, 并把结果存入表中. 因此, 权重中的一个 $g$-bit 分组就成为查表索引, 用于取得预计算结果. T-MAC 中的一个*表*用于保存一次 $[1,g]\times[g,2^g]$ 子矩阵乘法的结果, 表大小为 $[1,2^g]$. 在离线阶段, 1-bit 矩阵中的一个 tile 会连续存放在内存中, 便于快速加载. 与普通矩阵乘法的分块相同, 此处分块同样旨在改善 LUT 期间的数据局部性与 cache 利用率.

在线阶段给定 GEMM 的输入激活后, T-MAC 遍历激活中的每个 $[1,g]$ 向量, 将其与 $[g,2^g]$ 位模式矩阵相乘并建立查找表 (第 16 至 27 行). 查表时, 1-bit 权重矩阵的每个索引 (即分组) 都用于从表中取得部分结果 (第 6 至 9 行). 累加这些部分结果即可得到最终 GEMM 结果 (第 12 至 14 行).

举例来说, 令 $g=4$. 对于形状为 $[1,4]$ 的激活 $(A_1,A_2,A_3,A_4)$ 和形状为 $[4,M]$ 的 1-bit 权重, 激活会在线预计算为形状 $[1,16]$ 的 LUT, 其中包含从 $-A_1-A_2-A_3-A_4$ 到 $A_1+A_2+A_3+A_4$ 的各项. 每 4 个权重组成一组后, 权重向量 0000 会查得 $-A_1-A_2-A_3-A_4$, 0101 则会查得 $-A_1+A_2-A_3+A_4$.

<span id="figure-03"></span>

![图 3. T-MAC 与通用做法的数据流对比.](./t-mac/figure-03.png)

**图 3.** T-MAC 与通用做法的数据流对比.

**基于 LUT 的 mpGEMM 示例.** [图 3](#figure-03) 左侧给出了 T-MAC 在 CPU 上的具体实现示例. 这里取分组大小 $g=4$, 索引矩阵的 tile 大小 $W_i[K_{tk},M_{tm}]=[4,32]$, 位宽 $b=4$. 右侧是 llama.cpp 实现 mpGEMM 的通用做法. 左侧的 T-MAC 首先把 32 个 uint4 索引解包为 uint8 字节 (蓝色), 以兼容硬件数据类型和指令. 随后用这些 uint8 索引查表. 查表结果再被拆分并转换为更高精度, 用于乘以低比特 LLM 模型的量化 scale.

相比之下, 通用做法会为这个 4-bit 模型设计专用计算内核. 它先把 4-bit 权重直接解码为 int8, 使其与硬件数据类型对齐, 再对激活向量和权重向量执行 int8 点积. 同样, 结果会被转换为 FP16 后再乘以量化 scale. llama.cpp 的 cache 分块为 $W[K_{tk},M_{tm}]=[32,1]$. [第 3.2 节](#section-3-2) 将解释 T-MAC 与现有做法采用不同分块依据的原因.

**LUT 实现面临的挑战.** 从上述算法与示例可以看出, 基于 LUT 的 mpGEMM 有以下难点. *(i) 随机数据访问.* 现有做法进行连续数据访问, 而查找表需要按索引随机访问. 为降低访问成本, 必须把表放在高速片上内存中. *(ii) 片上内存用量增加.* 与现有做法相比, LUT 需要更多片上内存, 且表大小随分组大小 $g$ 指数增长. 例如 $g=4$ 时, LUT 比原始激活大 4 倍. 此外, 传统 GEMM 实现的每个基本块产生*标量*输出, LUT 方法产生的却是*向量*输出 (如 [图 3](#figure-03) 所示), 因而需要更多片上内存存储临时结果. 再看 [图 3](#figure-03) 的示例, LUT 方法使用 144 个 8-bit 寄存器, llama.cpp 使用 104 个 8-bit 寄存器.

针对上述问题, 我们提出两项主要技术: (a) *以 LUT 为中心的数据布局*, 将中间结果与 LUT 放入带宽更高的内存; (b) *压缩 LUT 存储*, 减小 LUT 大小并限制查表操作次数.

**算法 1: T-MAC GEMM**

- **输入:** 形状为 $N$, $K$ 的激活 $A$; 形状为 $M$, $K$ 的权重 $W$.
- **输出:** 形状为 $N$, $M$ 的结果矩阵 $R$.
- 令 $b$ 为权重位数.
- $W_1,\ldots,W_b\leftarrow\mathrm{PreprocessWeights}(W,M,K)$.
- $\mathrm{LUT}\leftarrow\mathrm{Precompute}(A,N,K)$.
- 令 $\alpha_i$ ($i\leftarrow 1$ 至 $b$) 为 bit-serial 的乘数.
- 令 $\beta$ 为 bit-serial 的偏置.
- **对于** $i\leftarrow 1$ 至 $b$:
  - **对于** $n,m\leftarrow 1$ 至 $N,M$:
    - $R_i[n,m]=\sum_{k\leftarrow 1}^{K}\operatorname{Lookup}(\mathrm{LUT},W_i,n,m,k)$.
- 令 $B$ 为形状 $M$, $K$ 且所有元素均为 $\beta$ 的矩阵.
- $R_\beta\leftarrow A\cdot B^\top$.
- $R\leftarrow\sum_{i\leftarrow 1}^{b}\alpha_iR_i+R_\beta$.
- **函数** $\mathrm{Precompute}(A,N,K)$:
  - 令 $g$ 为 LUT 的分组大小.
  - **对于** $n,k\leftarrow 1$ 至 $N,K$:
    - **对于** $i,j\leftarrow 1$ 至 $2^g,g$:
      - **如果** $i\mathbin{\texttt{\&}}(1\mathbin{\texttt{<<}}j)$:
        - $\mathrm{LUT}[n,k/g,i]\mathrel{+}=A[n,k]$.
      - **否则:**
        - $\mathrm{LUT}[n,k/g,i]\mathrel{-}=A[n,k]$.
  - **返回** $\mathrm{LUT}$.
- **函数** $\mathrm{PreprocessWeights}(W,M,K)$:
  - **对于** $i\leftarrow 0$ 至 $b$:
    - **对于** $m,k\leftarrow 1$ 至 $M,K$:
      - $W_i[m,k/g]\mathrel{+}=(W[m,k]\mathbin{\texttt{>>}}i)\mathbin{\texttt{<<}}(k\mathbin{\%}g)$.
  - **返回** $W_1,\ldots,W_b$.

<span id="section-3-2"></span>

### 3.2 以 LUT 为中心的数据布局

如 [第 3.1 节](#section-3-1) 所述, 基于 LUT 的低比特 GEMM 需要更多内存来保存查找表和中间结果. 另一方面, 随机内存访问往往使查表效率不高. 为解决存储和访问效率问题, 我们为基于 LUT 的低比特 GEMM 设计了以 LUT 为中心的数据布局. 具体而言, 它把查找表放在寄存器等片上内存中以加速访问, 并通过轴重排和数据分块增强数据复用, 从而降低内存用量. 为进一步提高效率, 我们还设计了两种数据布局优化: 用权重置换对齐内存事务, 用权重交错优化权重解包.

**把查找表放入片上内存.** [第 3.1 节](#section-3-1) 所述的 LUT GEMM 需要从预计算表中取结果, 因而会随机访问查找表. 为加速查表, 我们把查找表放在寄存器中, 并利用硬件专用指令 (如 ARM CPU 上的 TBL 和 x86 CPU 上的 PSHUF) 执行查表. [第 4 节](#section-4) 将说明如何用硬件专用指令优化查表. 不过, 把查找表放入寄存器会进一步增大片上内存压力, 内存溢出可能导致性能大幅下降. 为充分发挥片上内存查表的潜力, 我们用轴重排和分块增强数据复用, 从而减轻片上内存压力.

**轴重排.** 对 GEMM $C[N,M]=A[N,K]\times W[M,K]$ 而言, 通常先遍历空间轴 $N$ 和 $M$, 再遍历时间轴 $K$. 然而, LUT GEMM 需要沿 $K$ 轴建表. 若沿用传统 GEMM 先空间轴后时间轴的顺序, 就需要为整个 $A[N,K]$ 建立查找表, 存储量极大. 如果把轴顺序换成先时间轴后空间轴, 则只需维护一个小型 $[1,K]$ 查找表. 因此, T-MAC 先访问时间轴 $K$, 再访问空间轴 $N$ 与 $M$.

**分块.** 分块是一种常见技术, 它通过复用片上内存中的数据改善 GEMM 的数据局部性并降低内存需求. 假设 GEMM $C[N,M]=A[N,K]\times W[M,K]$ 用 tile $A[N_{tn},K_{tk}]$ 与 $W[M_{tm},K_{tk}]$ 处理, 每个 tile 只需从 DRAM 向处理器片上内存加载 $N_{tn}*K_{tk}+M_{tm}*K_{tk}$ 份数据, 而非 $N_{tn}*M_{tm}*K_{tk}$ 份. 在传统 GEMM 中, tile 大小 $N_{tn}$ 与 $M_{tm}$ 对效率的影响相同; $K_{tk}$ 不影响数据复用, 只需设置为与内存事务对齐.

但在 LUT GEMM 中, 激活 $A[N,K]$ 需要经过处理来建立查找表, 而权重 $W[M,K]$ 可以共享同一个预计算查找表. 换言之, $M$ 轴上更大的 tile 大小 $M_{tm}$ 能带来更充分的查找表复用. T-MAC 会综合考虑 $N_{tn}$, $M_{tm}$ 和 $K_{tk}$ 的分块配置, 以获得更好的数据复用.

<span id="figure-04"></span>

![图 4. 通过交错存放权重实现快速解包.](./t-mac/figure-04.png)

**图 4.** 通过交错存放权重实现快速解包.

**布局优化.** 除了通过轴重排与分块优化片上查表, 我们还设计了权重置换和权重交错两种数据布局优化, 进一步提高效率.

*用于连续内存访问的权重置换.* DRAM 需要连续访问才能获得更高的带宽利用率, 而基于分块的 GEMM 在访问 tile 时会产生随机访问, 因为输入矩阵的各个 tile 并未连续存储. 为此, T-MAC 置换权重矩阵, 使权重加载与内存事务对齐. 确定 GEMM 调度后, T-MAC 会重排输入矩阵, 让各个 tile 而不是整个矩阵按访问顺序连续存储. 具体来说, T-MAC 先依次展平一个 tile 中的元素, 再按照 tile 的访问顺序拼接这些展平结果. LLM 推理期间权重矩阵不会改变, 因此这种置换可在离线阶段完成, 不会增加推理开销.

*用于快速解包的权重交错.* 如 [第 3.1 节](#section-3-1) 所述, 权重矩阵以打包格式存于内存, 计算时需要解包. 现代 CPU 普遍采用小端序, 整数内的字节以反向顺序存储, 因而用整数指令解包权重后还需额外重排, 才能得到顺序正确的权重. 由于 LLM 推理期间权重矩阵不会改变, T-MAC 可以预先交错打包权重, 消除这一步重排. [图 4](#figure-04) 展示了一个示例: 对交错后的权重解包, 即可按顺序直接得到所需权重.

<span id="section-3-3"></span>

### 3.3 减少 LUT 存储

对基于 LUT 的低比特 LLM 推理来说, 查找表大小是同时影响存储需求与访问延迟的关键因素, 在 [第 3.2 节](#section-3-2) 以片上内存优化查表时尤其如此. 表越大, 占用的内存空间越多, 访问速度也越慢. 为解决这一问题, 我们引入两种优化策略: *镜像合并*与*查找表量化*. 如 [图 5](#figure-05) 所示, 镜像合并利用表值的对称性把表长减半, 查找表量化则量化表值本身来缩短每项的位宽. 两者结合后, 可以在不损失 LLM 推理精度的前提下显著缩小查找表的存储空间, 最多降至原来的四分之一.

<span id="figure-05"></span>

![图 5. 通过镜像合并和查找表量化减少 LUT 存储. 镜像合并将表长减半, 查找表量化缩短表值位宽.](./t-mac/figure-05.png)

**图 5.** 通过镜像合并和查找表量化减少 LUT 存储. 镜像合并将表长减半, 查找表量化缩短表值位宽.

**镜像合并.** LLM 推理查找表中的数值天然具有对称性, 为优化提供了独特机会. 表中每个正值都会与其负值成对出现, 两者关于零值镜像对称. 我们提出的*镜像合并*利用这一性质, 只显式保存一半表值, 另一半只需对已存值取反即可快速重建. 这种表压缩方法无损, 能完整保持模型推理精度. 它也很高效, 可加速查找表的预计算, 减少存储需求并加快查表.

**查找表量化.** 查找表量化的原理与权重和激活量化相似, 通过降低表值精度来提高计算效率. 例如, 查找表中原本以 16-bit 浮点数 (fp16) 表示的值, 可以结合 scale 量化为 8-bit 整数 (int8). 查找表量化对模型精度的影响可以忽略. 为保证计算速度, 激活量化往往必须采用粗粒度和静态量化, 因而难以维持模型精度; 与之不同, 我们采用更细的粒度 ($k=4$ 时量化 8 个值) 和动态量化, 尽量减少精度下降. [第 5.6 节](#section-5-6) 的结果表明, 查找表量化对模型整体精度的影响几乎无法察觉. 就效率而言, 它显著降低了查找表的存储需求, 并加快了查表过程.

<span id="section-4"></span>

## 4 实现

<span id="table-01"></span>

![表 1. 用于查表和聚合的硬件 intrinsic.](./t-mac/table-01.png)

**表 1.** 用于查表和聚合的硬件 intrinsic.

**通过 TVM 生成代码.** 我们使用 TVM [Che18d] + LLVM [Lat04] 生成代码. 这样既能针对不同形状的 GEMM 和不同硬件生成最优代码, 也能实现循环展开, 向量化和常量折叠等常见优化. 我们用 TVM Tensorize 把硬件 intrinsic 嵌入代码, 并用 AutoTVM [Che18a] 针对不同硬件目标自动微调生成的代码.

**API 与集成.** 我们为 C++ 和 Python 提供一致的 API. GEMM 函数封装为 TVM PackedFunc, tensor 可通过 DLPack [Dlp17] tensor 结构传递, 因而能够方便地与 PyTorch, NumPy 等框架互操作. 此外, 我们还提供一个 C++ wrapper, 可通过原始指针传递 tensor, 并去除对 TVM runtime 的依赖. 这为集成到其他 C++ 项目提供了更轻量的方案.

**并行化.** 我们用 TVM runtime thread pool 向 CPU 动态分配任务. 但在把 T-MAC 集成到 llama.cpp 时, 我们发现 llama.cpp thread pool 与 TVM thread pool 存在明显冲突. 两个 thread pool 的线程争用 CPU 资源, 导致性能显著下降. 为解决该问题, 我们不让 TVM 直接生成 library 文件, 而是生成 C++ 代码, 再后处理代码以去除对 TVM runtime 和 thread pool 的依赖. 生成的函数只执行单个 thread block 的计算, 随后由我们把这些 thread block 分配给 llama.cpp thread pool 中的不同线程. 这种方法带来了更好的性能和 llama.cpp 兼容性. 可移植的 C++ 代码也为跨平台部署提供了选择.

**用 TBL/PSHUF 高效查表.** 把表载入寄存器后, 我们可以使用 ARM NEON/Intel AVX2 提供的硬件 intrinsic. NEON 和 AVX2 都提供 8-bit 查表指令. ARM NEON 的位宽为 128, 恰好容纳 $g=4$ 的整张表. Intel AVX2 的位宽为 256, 但上下两半位于两个独立的 128-bit lane. 因此, 我们复制查找表以填满 256-bit LUT 寄存器, 用一条指令查询 32 个不同的 int8 权重索引. 如果表的数据类型为 float16, 由于 NEON/AVX2 不支持 16-bit LUT, 我们会把 float16 拆成两个 8-bit LUT, 分别保存低位与高位. 用两条指令分别查询两部分后, 再将其合成为 float16.

**确定片上 LUT 的大小.** 我们针对每种硬件调节片上 LUT 数量, 确保充分利用片上内存, 且处理 tile 时 LUT 不会被换出. 更大的片上内存可容纳更多 LUT, 因而能在回写前聚合更多中间结果. 片上 LUT 过多会导致寄存器溢出并产生额外开销.

$g$ 也由片上内存大小和指令吞吐量决定. 对 ARM.TBL/AVX2.PSHUF 而言, $g=4$ 的 LUT 恰好装入一个寄存器. 更大的 $g$, 如 $g=5$, 需要两个寄存器以及速度更慢的 ARM.TBL2/AVX512.PSHUF.

**快速 8-bit 聚合.** 除查表外, 聚合也是一项显著的计算开销. 为加速聚合, 我们先以低位宽聚合查表结果, 再把聚合和转换为 float16 等更高精度, 不会损失精度. 如果表被量化为 int8, 还可以实现快速 8-bit 聚合 [Bla21a]. 通常应把 int8 值转换为 int16 以避免溢出, 但 int16 指令的吞吐量只有 int8 聚合的一半. 另一种做法是用 *avg/rhadd* 指令计算平均值, 再从最终值中减去概率偏差, 尽量降低精度损失. 需要注意, 快速 8-bit 聚合可能造成不可忽略的精度损失. [表 1](#table-01) 列出了不同 CPU 架构上用于查表与聚合的硬件 intrinsic.

**Bit-serial 线性变换.** 在 [第 3.1 节](#section-3-1) 的分解中, 我们使用 $v_i$ 的原始值, 即 0 和 1. 不过, 也可以对这些值引入线性变换. 将该线性变换记为 $f(v_i)$, 变换后的值记为 $f(0)=s_0$ 和 $f(1)=s_1$.

为加速预计算并减少量化误差, 需要谨慎选择 $s_0$ 和 $s_1$. 为避开浮点乘法指令, 我们从集合 $[-1,0,1]$ 中选择它们. 为降低量化误差, 我们尽量缩小查找表中最大值与最小值之差. 实验表明, $s_0=-1$ 和 $s_1=1$ 是最优选择.

设定 $s_0$ 与 $s_1$ 后即可定义线性变换 $f$, 此时 $W$ 的分解应调整如下:

$$
\begin{aligned}
f(v_{i}) & =\alpha_{i}^{\prime}v_{i}+\beta_{i}^{\prime},v_{i}=\alpha_{i}f(v_{i})+\beta_{i}, \\
\mathrm{where}~\alpha_{i}=\frac{1}{\alpha_{i}^{\prime}},\beta=-\frac{\beta_{i}^{\prime}}{\alpha_{i}^{\prime}} \\
W & =\sum_{i=0}^{b-1}\alpha_{i}2^{i}W_{i}^{\prime}+B, \\
\mathrm{where}~W_{i}^{\prime}=f(W_{i}),B=J\cdot\sum_{i=0}^{b-1}\beta_{i}2^{i}
\end{aligned}
$$

**用寄存器 swizzling 高效预计算 LUT.** 如 [第 3.1 节](#section-3-1) 所示, 我们选择加减指令而非乘法指令来获得更高吞吐量. 对形状为 $(N,K/g,2^g)$ 的 LUT, 加减运算可沿 $K/g$ 轴向量化. 例如:

$$
\begin{aligned}
\mathrm{LUT}[0,0:8,0]= & -A[0,0:32:4]-A[0,1:32:4] \\
-A[0,2:32:4]-A[0,3:32:4]
\end{aligned}
$$

对 $A$ 的索引并不连续. 借助 NEON 的 *LD4* 和 AVX2 的 *vgatherdps*, 可以高效加载不连续数据. 但在把不连续的 LUT 写回内存时, 对 AVX2 而言, 从 SIMD 寄存器提取特定字节再写入内存效率很低. 为解决这个问题, 我们用寄存器 swizzling 重排 LUT, 使其可以连续回写内存. 首先用 *vpblendvb* 把不同寄存器中的 8-bit 值混合到一个寄存器, 随后用 *vpermd* 重排 256-bit 寄存器中的 32-bit 值, 再用 *vpshufb* 把 8-bit 值进一步排列到正确顺序. 完成 swizzling 后, LUT 即可连续写回内存.

<span id="section-5"></span>

## 5 评估

<span id="table-02"></span>

![表 2. 硬件设备规格.](./t-mac/table-02.png)

**表 2.** 硬件设备规格.

我们在 4 种不同的边缘设备上, 用真实的低比特 LLM, 具体为 Llama 和 BitNet, 评估 T-MAC. 基准测试旨在与当前先进的 *llama.cpp* 实现直接比较. 主要发现如下:

- T-MAC 的 mpGEMV 和 mpGEMM 内核性能明显提升, 大幅超过当前先进的反量化内核.
- 与原始 *llama.cpp* 实现相比, T-MAC 将端到端模型推理吞吐量提升 2-4 倍, 同时把能耗降低 60%-70%.
- 在许多情况下, T-MAC 的性能不仅追平 GPU, 甚至超过 GPU, 为设备端 LLM 部署效率树立了新的里程碑.

<span id="section-5-1"></span>

### 5.1 评估设置

<span id="figure-06"></span>

![图 6. 单线程和多线程下 1/2/3/4 bit mpGEMV 性能基准. 矩阵形状取自 Llama-2-7B 和 Llama-2-13B. llama.cpp 的 1-bit 内核性能由其 2-bit 内核推算, 以虚线标出.](./t-mac/figure-06.png)

**图 6.** 单线程和多线程下 1/2/3/4 bit mpGEMV 性能基准. 矩阵形状取自 Llama-2-7B 和 Llama-2-13B. llama.cpp 的 1-bit 内核性能由其 2-bit 内核推算, 以虚线标出.

**硬件设备.** 如 [表 2](#table-02) 所示, 我们在 4 种不同的边缘设备上评估 T-MAC, 从高性能的 M2-Ultra 到算力较弱的 Raspberry Pi 均包含在内. 受测 CPU 涵盖 Intel Core, Apple Silicon 和 Cortex 系列, 操作系统包括 OSX, Linux 和 Windows. 这项评估验证了 T-MAC 的跨平台兼容性, 以及它在不同指令集和多种边缘部署场景中的稳定性能.

**内核与模型.** 为评估 T-MAC 性能, 我们用真实的低比特 LLM 和实际场景进行了大量基准测试. 内核性能测试采用源自 Llama-2-7B 与 Llama-2-13B 的矩阵形状, 确保评估反映实际需求. 端到端吞吐量测试采用真实量化模型, 展示 T-MAC 在不同位宽配置下的实际效果. 具体而言, 我们使用 4-bit, 3-bit, 2-bit 和 1-bit 量化 Llama 模型, 以及从头训练的 1-bit 和 1.58-bit BitNet 模型. 1.58-bit BitNet 中的三值权重按 2-bit 处理, 并拆成两个 1-bit 矩阵. 4-bit Llama 模型来自 GPTQ [Fra22], 3-bit 和 2-bit 模型来自 BitDistiller [Du24], 1-bit 模型来自 OneBit [Xu24h].

**基线.** 我们将 T-MAC 与面向边缘设备 LLM 部署的先进实现 *llama.cpp* (版本 b2794, 发布于 2024 年 5 月) 比较. 选择 *llama.cpp* 作为基线有几个重要原因. 首先, *llama.cpp* 代表边缘设备 LLM 部署的前沿水平, 针对每种硬件平台都配有高度优化的内核. 它用途广泛且性能稳健, 很适合用于评估新方法. 此外, *llama.cpp* 采用无依赖的纯 C/C++ 实现, 在多种硬件配置上具有很高的兼容性与效率. 内核性能测试以 *llama.cpp* 为相应硬件提供的优化内核为基线. 在端到端吞吐量评估中, 我们把 T-MAC 的 LUT 内核集成到 *llama.cpp*, 再与原始 *llama.cpp* 比较.

我们还将 T-MAC 与 *llama.cpp (BLAS)* 比较. *llama.cpp* 在 M2-Ultra 上使用 *Accelerate*, 在其他平台使用 *OpenBLAS*. 与 *llama.cpp* 高度优化的混合精度实现相比, *llama.cpp (BLAS)* 的 mpGEMV 较慢, mpGEMM 较快. 因此, 只有 mpGEMM 将 T-MAC 与 *BLAS* 比较.

**测量方法.** 我们同时进行*内核级*与*模型级*测量. 为在 CPU 上获得准确且稳定的内核级延迟, 先预热 10 次, 再运行 100 次并取平均值. M2-Ultra 的预热略有不同, 至少需要 1 秒才能达到最高性能. 测量模型级延迟时, 我们把 T-MAC 集成到 llama.cpp, 重复 20 轮生成 64 个 token, 以评估 token 生成吞吐量.

<span id="figure-07"></span>

![图 7. 多线程下 1/2/3/4 bit mpGEMM 性能基准. 矩阵形状取自输入序列长度为 256 的 Llama-2-7B 和 Llama-2-13B. llama.cpp 的 1-bit 内核性能由其 2-bit 内核推算, 以虚线标出.](./t-mac/figure-07.png)

**图 7.** 多线程下 1/2/3/4 bit mpGEMM 性能基准. 矩阵形状取自输入序列长度为 256 的 Llama-2-7B 和 Llama-2-13B. llama.cpp 的 1-bit 内核性能由其 2-bit 内核推算, 以虚线标出.

<span id="section-5-2"></span>

### 5.2 mpGEMV/mpGEMM 性能基准

我们在全部 4 台设备上评估 Llama-2-7B/13B 的内核. 如 [图 6](#figure-06) 所示, 位宽从 4-bit 降至 2-bit 时, llama.cpp 没有获得额外加速; 由于解码开销, 3-bit 甚至比 4-bit 慢 15%. 因而即使 llama.cpp 没有 1-bit 实现, 也可以推断其 1-bit 性能与 2-bit 相近. 相比之下, T-MAC 随位宽降低获得线性加速. 对单线程 GEMV, T-MAC 在 1/2/3/4 bit 下的最高加速比分别为 11.2x, 5.8x, 4.7x 和 3.1x. 多线程 mpGEMV 的 T-MAC 性能主要受内存带宽限制, 但高效的内存访问仍能带来显著加速. 例如在 2-bit 下, T-MAC 在 4 台设备上的加速比分别为 4.0x, 4.0x, 5.31 和 2.5x.

[图 7](#figure-07) 评估了序列长度为 256 时的多线程 mpGEMM. *llama.cpp* 使用 BLAS 执行 mpGEMM. 在 2-bit 下, T-MAC 仍在 RBP, Orin 和 Surface 上取得显著加速, 最高加速比分别为 4.0x, 5.3x 和 5.3x. M2-Ultra 是个例外, 因为 Apple Silicon CPU 配有强大的 AMX 协处理器来处理 GEMM. 即便如此, T-MAC 在 1-bit 下仍取得最高 $2.0\times$ 加速.

T-MAC 在 3-bit 精度下优势明显, 原因是现有技术处理 3-bit 权重的效率较低. 权重解码通常用 SHIFT 和 AND 指令完成, 要求权重与字节宽度对齐. 8 无法被 3 整除, 因而解码效率很低. *llama.cpp* 尝试把 2 bit 与剩余 1 bit 分开打包来优化, 但仍有显著开销. T-MAC 分别计算每个 bit 的结果, 从而避开这一问题.

<span id="section-5-3"></span>

### 5.3 端到端推理吞吐量

集成到 llama.cpp 后, 我们比较 llama.cpp 与 T-MAC 的端到端 token 生成吞吐量. BitNet 以 2-bit 执行. 如 [图 8](#figure-08) 所示, 在 Raspberry Pi 5 单线程运行时, T-MAC 对 3 个模型分别取得 2.8x, 6.7x 和 5.8x 加速. 多线程下受内存限制及 mpGEMV/mpGEMM 以外的算子影响, 加速不那么明显. 不过在 M2-Ultra 上仍观察到 1.1x, 2.3x 和 1.7x 加速. T-MAC 在高性能 M2-Ultra 上最高可达 71 tokens/s, 在算力较弱的 Raspberry Pi 5 上可达 11 tokens/s, 显示出实际边缘部署的良好前景.

<span id="figure-08"></span>

![图 8. 把 T-MAC 内核集成到 llama.cpp 后的端到端 token 生成吞吐量. M1, M2 和 M3 分别表示 Llama-2-7B-4bit, Llama-2-7B-2bit 和 BitNet-3B.](./t-mac/figure-08.png)

**图 8.** 把 T-MAC 内核集成到 llama.cpp 后的端到端 token 生成吞吐量. M1, M2 和 M3 分别表示 Llama-2-7B-4bit, Llama-2-7B-2bit 和 BitNet-3B.

<span id="section-5-4"></span>

### 5.4 功率与能耗

除计算效率外, 能源效率同样关键, 对依赖电池供电的边缘设备尤其如此. 为评估 T-MAC 相对 *llama.cpp* 的功率与能耗, 我们在 M2 Ultra 上用多线程实现进行实验. 分析选用 3 个模型: Llama-2-7B-4bit, Llama-2-7B-2bit 和 BitNet-3B. 功率由 OSX 上的 *powermetrics* 测量, 它可记录指定采样间隔内的平均功率. 我们把间隔设为 500 ms, 连续生成 token 至少 120 s, 再对功率随时间积分以计算总能耗.

如 [图 9](#figure-09) 所示, 使用 T-MAC 的 LUT 内核可显著降低功率. 对 Llama-2-7B-4bit, Llama-2-7B-2bit 和 BitNet-3B, 功率分别降低 10.3%, 10.3% 和 17.3%. 功率下降再结合 T-MAC 带来的延迟改善, 使总能耗大幅减少. 具体而言, 3 个模型的能耗分别降低 20.6%, 61.2% 和 51.3%.

<span id="figure-09"></span>

![图 9. M2-Ultra 多线程推理的功率与能耗. M1, M2 和 M3 分别表示 Llama-2-7B-4bit, Llama-2-7B-2bit 和 BitNet-3B.](./t-mac/figure-09.png)

**图 9.** M2-Ultra 多线程推理的功率与能耗. M1, M2 和 M3 分别表示 Llama-2-7B-4bit, Llama-2-7B-2bit 和 BitNet-3B.

<span id="section-5-5"></span>

### 5.5 优化分解

为评估 [第 3 节](#section-3) 中各项优化的效果, 我们逐项拆解优化策略. 这些优化多数对单线程帮助更大, 但分块需要多线程才能生效, 因此本项评估采用多线程. 大部分优化彼此正交, 但有些依赖其他优化, 例如置换必须在分块完成后进行. 因此, 我们从基础实现 *TM-base* 开始, 逐步加入各项优化.

*TM-base* 用硬件 intrinsic 加速查表, 但没有实现任何内存访问优化. 如 [图 10](#figure-10) 所示, 它最多比 llama.cpp 基线慢 17%. 加入*查找表量化*后, 性能已可与 llama.cpp 竞争. *分块*优化进一步带来最高 1.45x 加速. *置换*把每个 tile 的数据布局重排为连续内存, 又带来额外 1.39x 加速. 图中*调优*的效果似乎不明显, 因为默认分块配置已与 M2-Ultra 的寄存器和 cache 很好地对齐; 但在其他设备上, 调优有助于找到更优配置.

加入*权重交错*后便得到 T-MAC. 交错消除了大部分解包开销, 带来 1.42x 加速. 激进的*快速聚合*最多还能让 T-MAC 加速 1.29x, 但可能造成不可忽略的精度损失, 因此我们把它作为可选优化.

<span id="figure-10"></span>

![图 10. 在 M2-Ultra 上逐步应用 T-MAC 优化后 Llama-2-7B/13B GEMV 内核的多线程性能. S0-S5 是图 6 所示的不同形状. TM: T-MAC, TQ: 查找表量化, Perm.: 置换, IL: 交错, FA: 快速聚合.](./t-mac/figure-10.png)

**图 10.** 在 M2-Ultra 上逐步应用 T-MAC 优化后 Llama-2-7B/13B GEMV 内核的多线程性能. S0-S5 是 [图 6](#figure-06) 所示的不同形状. TM: T-MAC, TQ: 查找表量化, Perm.: 置换, IL: 交错, FA: 快速聚合.

<span id="section-5-6"></span>

### 5.6 误差分析

与传统 mpGEMM 实现相比, 误差有两个来源: (a) *查找表量化*, 即本方法包含的算法近似; (b) *快速聚合*, 其误差来自固定 CPU 架构中的指令执行. 我们分别在内核级与模型级评估两种误差来源的影响.

**内核级评估.** 我们以未量化的 $W_{\mathrm{FP}16}A_{\mathrm{FP}16}$ GEMV 为基准. GEMV 的权重和激活是服从*高斯分布*的随机 FP16 值, 随后量化为 4-bit, 交由 llama.cpp 和 T-MAC 执行. 再计算 ground truth 与 mpGEMV 输出之间的归一化均方误差 (NMSE). 如 [表 3](#table-03) 所示, llama.cpp 与 T-MAC 的 NMSE 差异可以忽略, 说明查找表量化误差很小. 但应用*快速聚合*后, NMSE 增至 $2.5\times$.

<span id="table-03"></span>

![表 3. 相对于未量化 ($W_{\mathrm{FP16}}A_{\mathrm{FP16}}$) GEMV 内核的 NMSE 误差.](./t-mac/table-03.png)

**表 3.** 相对于未量化 ($W_{\mathrm{FP16}}A_{\mathrm{FP16}}$) GEMV 内核的 NMSE 误差.

<span id="table-04"></span>

![表 4. M2-Ultra 单线程运行 Llama-2-7B-4bit 时的端到端吞吐量与模型质量. 在模型质量相同的情况下, T-MAC 比 llama.cpp 将吞吐量提升 $1.3\times$. 快速聚合 (FA) 可把吞吐量增益进一步提高到 $1.6\times$, 但当前 CPU 指令的数值误差会使模型质量下降.](./t-mac/table-04.png)

**表 4.** M2-Ultra 单线程运行 Llama-2-7B-4bit 时的端到端吞吐量与模型质量. 在模型质量相同的情况下, T-MAC 比 llama.cpp 将吞吐量提升 $1.3\times$. 快速聚合 (FA) 可把吞吐量增益进一步提高到 $1.6\times$, 但当前 CPU 指令的数值误差会使模型质量下降.

**模型级评估.** 为考察这些误差对真实模型的影响, 我们选择 Llama-2-7B 测试. 未量化 ground truth 使用由官方 Llama-2-7B 权重转换的 GGUF 模型, mpGEMM 使用随 llama.cpp 发布的原始 llama-2-7b.Q4_0.gguf 模型 [Lla23b]. 把 T-MAC 集成到 llama.cpp 后, 我们通过 llama.cpp 提供的 *perplexity* [Lla23c] 工具评估. 评估包含 3 项任务: 在 *WikiText-2* [Mer16a] 和 *lambada_openai* [Pap16, Rad19] 上测量 perplexity (越低越好), 在 *WinoGrande* [Sak19] 上测量问答准确率 (越高越好). 如 [表 4](#table-04) 所示, T-MAC 在 3 项任务上的结果都与 llama.cpp 相同, 表明 T-MAC 引入的误差对真实模型可以忽略. 启用*快速聚合*后, 两项 perplexity 分别增加 0.4 和 1.0, 准确率下降 0.3%.

总之, T-MAC 在显著加速的同时只给模型推理引入可忽略的误差. *快速聚合*可以进一步提高性能, 代价是模型质量下降. 对重视实时性能且对精度不太敏感的场景, 我们把它作为可选项提供. 即使不用*快速聚合*, [图 10](#figure-10) 也表明 T-MAC 仍能取得可观收益. 我们预计, 未来只需对 CPU 微架构做直接优化, 便可缓解快速聚合引入的误差.

<span id="section-5-7"></span>

### 5.7 与 GPU/NPU 比较

GPU 广泛用于 LLM 部署. 为展示 T-MAC 的效率, 我们将 CPU 上的 T-MAC 与 GPU 上的 llama.cpp 比较. llama.cpp 是边缘设备上同时面向 CPU 和 GPU 的先进 LLM 推理框架.

<span id="figure-11"></span>

![图 11. NVIDIA Jetson AGX Orin 上 T-MAC (CPU) 与 llama.cpp (GPU) 的 mpGEMV 内核性能.](./t-mac/figure-11.png)

**图 11.** NVIDIA Jetson AGX Orin 上 T-MAC (CPU) 与 llama.cpp (GPU) 的 mpGEMV 内核性能.

[图 11](#figure-11) 比较了 T-MAC (CPU) 与 llama.cpp (GPU) 在 NVIDIA Jetson AGX Orin 上的 mpGEMV 内核性能. 该平台配有 ARM CPU 和 NVIDIA CUDA GPU, 所有内核配置均取自 Llama-2-7B. 在全部 W1A16 测试中, T-MAC 都显著超过 GPU; 在 W2A16 和 W3A16 下, 两者性能相当. 凭借强大的并行计算能力, GPU 在更高位宽和更大形状下表现更好, 但这项评估仍显示出 CPU 边缘设备 LLM 部署的巨大潜力.

<span id="table-05"></span>

![表 5. NVIDIA Jetson AGX Orin 上 Llama-2-7B-2bit 的端到端推理吞吐量, 功率和能耗对比.](./t-mac/table-05.png)

**表 5.** NVIDIA Jetson AGX Orin 上 Llama-2-7B-2bit 的端到端推理吞吐量, 功率和能耗对比.

<span id="table-06"></span>

![表 6. 受测平台的详细 CPU/GPU/NPU 规格. "Used Cores" 指充分利用内存带宽时使用的 CPU 核心数.](./t-mac/table-06.png)

**表 6.** 受测平台的详细 CPU/GPU/NPU 规格. "Used Cores" 指充分利用内存带宽时使用的 CPU 核心数.

<span id="table-07"></span>

![表 7. 3 台设备上, T-MAC 与 GPU/NPU 运行 Llama-2-7B-4bit/2bit 时的 token 生成速度 (tokens/s). NPU 的 2-bit 性能由 4-bit 推算, 以 "*" 标出.](./t-mac/table-07.png)

**表 7.** 3 台设备上, T-MAC 与 GPU/NPU 运行 Llama-2-7B-4bit/2bit 时的 token 生成速度 (tokens/s). NPU 的 2-bit 性能由 4-bit 推算, 以 "*" 标出.

[表 5](#table-05) 给出了 NVIDIA Jetson AGX Orin 上 Llama-2-7B-2bit 模型的端到端对比. 不用 T-MAC 时, CPU 只在功率方面优于 GPU; 由于吞吐量较低, 能耗仍不如 GPU. 与 CPU 上的 llama.cpp 相比, T-MAC 不仅把吞吐量提高到 $2.2\times$, 还把功率降至 69$\%$, 能源效率因而达到 $3.2\times$. 与 GPU 上的 llama.cpp 相比, T-MAC 的吞吐量虽只有 78$\%$, 所需功率却只有 34$\%$, 能源效率达到 $2.3\times$. [图 11](#figure-11) 显示 T-MAC 的 mpGEMV 内核超过 GPU. T-MAC 的端到端吞吐量仍低于 GPU, 原因在于 CPU 上 llama.cpp 除 mpGEMV 外的其他内核性能.

除了功率效率, T-MAC 在常用平台上的性能也超过 GPU/NPU. 我们进一步在 Surface Laptop 7, OnePlus 12 和 Jetson Orin NX 这 3 台设备上评估 T-MAC. 所配 CPU/GPU/NPU 的完整规格见 [表 6](#table-06). 我们采用能够充分利用内存带宽并接近最优性能的最少 CPU 核心数. GPU 评估中, NVIDIA GPU 使用 llama.cpp CUDA backend, Qualcomm GPU 使用 OpenCL backend. NPU 性能取自 Qualcomm 通过 Qualcomm AI Hub [Qua24] 发布的官方数据.

借助 T-MAC, CPU 可以获得高得多的计算吞吐量, 并充分利用内存带宽. 在大多数边缘设备上, GPU/NPU 与 CPU 共享统一内存, token 生成期间的 GEMV 同样受内存限制. 如 [表 7](#table-07) 所示, T-MAC 在 3 台设备上运行 Llama-2-7B-2bit 时都取得显著加速. 具体而言, Surface Laptop 7 上仅用总共 12 个 CPU 核心中的 4 个, T-MAC 就比 NPU 快 $3\times$; 在 OnePlus 12 上比 NPU 快 $1.5\times$; 在 Jetson Orin NX 上比 Ampere GPU 快 $1.4\times$. 即使运行 Llama-2-7B-4bit, T-MAC 在 Surface Laptop 7 上仍保持 $2.1\times$ 加速. 尤其在 OnePlus 12 上, T-MAC 运行 4-bit 与 2-bit 模型时分别比 Adreno GPU 快 $6.4\times$ 和 $9.7\times$.

总之, T-MAC 利用随处可得的 CPU, 相比 GPU 乃至专为 AI 工作负载设计的 NPU 都有明显性能优势. 这使 T-MAC 成为在边缘设备上部署 LLM 的实用方案.

<span id="section-6"></span>

## 6 相关工作

**LLM 量化算法.** LLM 量化已成为在资源受限环境中高效部署 LLM 的关键技术. 一部分研究致力于同时量化权重与激活. *LLM.int8()* [Det22] 把异常特征维度隔离出来用 16-bit 计算, 其余大多数维度则采用高效的 8-bit 计算. *SmoothQuant* [Xia23] 通过数学等价变换把量化难点从激活迁移到权重, 从而实现权重和激活的 INT8 量化. 随着研究推进, 关注点逐渐转向只量化模型权重, 因为权重存储占据了大部分内存. *GPTQ* [Fra22] 和 *AWQ* [Lin23d] 等算法已证明, 后训练量化可以把 LLM 量化到 4 bit. *BitDistiller* [Du24] 进一步结合量化感知训练 (QAT) 与自蒸馏, 把边界推进到 2 bit. *BitNet* [Wan23] 则采取更大胆的路线, 从头训练 1-bit LLM.

**LLM 推理系统.** LLM 的重要性推动了多种推理系统的发展, 它们面向不同平台, 并针对特定目标优化. *vLLM* [Kwo23a] 是专为 LLM 设计的高吞吐, 高内存效率推理引擎, 擅长大 batch 处理. *llama.cpp* [Lla23a] 采用无外部依赖的纯 C/C++ 实现, 在边缘计算设备上性能出色. *TensorRT-LLM* [Ten23] 集成了一系列专门面向 NVIDIA GPU 的先进优化. *Intel Neural Compressor* [Int18] 提供开源 Python 模型压缩库, 旨在提高 Intel 生态中 LLM 的效率. 这些推理系统都有一项关键能力: 支持低比特 LLM, 不仅减少内存用量, 还提高计算效率, 从而让更多应用能够使用 LLM. 除端到端 LLM 推理系统外, 也有工作专注于为低比特 LLM 开发高效计算内核 [Fra24, Bit24a].

<span id="section-7"></span>

## 7 结论

T-MAC 把以数据类型为中心的乘法转换为逐比特查表, 为日益普及的 mpGEMM 提供统一且可扩展的方案. 在边缘设备普遍配备的 CPU 上, T-MAC 内核相比 llama.cpp 最高可加速 $6.6\times$, 使 CPU 推理速度能够追平甚至超过同一设备上的 GPU. 因此, T-MAC 为不依赖 GPU 的边缘设备 LLM 部署提供了实用方案, 即使 Raspberry Pi 也不例外. LUT 的硬件实现效率远高于乘法, T-MAC 也由此为基于 LUT 设计新型 LLM 硬件加速器开辟了广阔空间.

[+1]: W# 表示权重位宽, A# 表示激活位宽.
