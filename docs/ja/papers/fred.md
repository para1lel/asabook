---
title: 'FRED'
createTime: 2026/08/24 20:00:00
permalink: /ja/papers/fred/
pageClass: paper-reading
---

> [Saeed Rashidi](https://orcid.org/0000-0002-6472-9920), [William Won](https://orcid.org/0000-0002-1715-9144), [Sudarshan Srinivasan](https://orcid.org/0009-0002-8662-5820), [Puneet Gupta](https://orcid.org/0000-0002-6188-1134), and [Tushar Krishna](https://orcid.org/0001-5738-6942). 2024 年 6 月 28 日に arXiv へ初投稿、現行版は v2 (2025 年 6 月 9 日). [第 52 回 IEEE/ACM International Symposium on Computer Architecture (ISCA 2025)](https://doi.org/10.1145/3695053.3731055) 掲載、2025 年 6 月 21-25 日. [FRED: A Wafer-scale Fabric for 3D Parallel DNN Training](https://arxiv.org/abs/2406.19580v2). [原著 PDF](/paper/fred.pdf). [arXiv DOI](https://doi.org/10.48550/arXiv.2406.19580). [TeX ソース](https://export.arxiv.org/e-print/2406.19580v2). 正確な印刷レイアウトと参考文献は原著 PDF を参照されたい.

## 概要

ウェハスケールシステムは, 高性能アクセラレータチップレットと高速ウェハスケールインターコネクトを密に統合し, 低レイテンシかつ広帯域の接続を提供する技術である. これは深層ニューラルネットワーク (DNN) の学習基盤として有望である. しかし, 2D Mesh など現在のウェハ上ネットワークトポロジーは, さまざまな並列化戦略を効率よく扱うための柔軟性に欠ける. 本稿では, DNN 学習の通信要件に合わせたウェハスケールファブリック Fred を提案する. Fred は小型マイクロスイッチによる分散ウェハ上トポロジーを構成し, 任意のアクセラレータ群の集合通信に対するノンブロッキング接続とスイッチ内集合通信を実現する. 例示した並列化戦略では, ベースラインのウェハスケール Mesh と比べて, ResNet-152, Transformer-17B, GPT-3, Transformer-1T のエンドツーエンド学習時間を平均でそれぞれ $1.76\times$, $1.87\times$, $1.34\times$, $1.4\times$ に短縮できる.

<span id="section-1"></span>

## 1 序論

DNN models are on an exponential growth curve. A recent study shows that in less than two years, the compute and memory requirements for DNN training have increased by 1,$800\times$ and 1,$500\times$, respectively [Thi21]. Distributing the training across multiple accelerators or neural processing units (NPUs) is a common practice today to reduce the training time. However, one critical side effect of distributed training is the communication overhead between NPUs to synchronize model gradients and/or activations, depending on the parallelization strategy. As the number of NPUs scales, communication overhead increases, up to a point where it becomes the dominant factor in distributed training latency [Ast20, Sca20b, Jia19a, An20].

There are fundamental limits to the bandwidth that can be provided even by high-speed rack-scale fabrics (such as NVLink [Eva19]), and thus there has been a growing interest in platforms that integrate multiple NPUs together in the same package. Cerebras [Cer21] demonstrated one extreme incarnation of this idea in the form of a monolithic wafer with NPUs connected to one another. More cost- and yield-effective approaches include silicon/organic interposer-based approaches [Sim19, Cen20] or using Silicon Interconnect Fabric (Si-IF), which bonds chiplets directly onto a full thickness silicon *wafer* without needing a package [Arc19, Des21, Waf24]. *In this work, we assume a passive, interconnect-only wafer-scale substrate onto which chiplets are bonded at fine pitch similar to Si-IF or TSMC-SoW [Tsm24]. This allows for heterogeneous integration of compute, memory, and network chiplets from disparate technologies, unlike the monolithic Cerebras approach.*

While there is broad agreement on the scalability and bandwidth benefits that wafer-scale substrates can provide, the *architecture of the fabric connecting the NPUs* remains an open question. All wafer-scale accelerator proposals to date (e.g., Cerebras CS2 [Cer21], NVIDIA’s SIMBA [Sim19], UCLA’s waferscale GPU [Arc19], Chiplet Cloud [Chi24a], Chen et al., TTO [Enh24]) have implemented a 2D Mesh topology for the fabric. The choice of a Mesh is understandable. It is the most pervasive topology in many-core chips given its ease of place-and-route and scalability and is the natural choice on a wafer-scale substrate as well. *However, we demonstrate that the inherent blocking nature of the 2D Mesh topology is extremely inefficient for DNN training communication use cases*.

<span id="figure-01"></span>

![図 1. HW-SW Co-Design Stack for Optimizing DNN training communication. This work addresses the three phases highlighted in red. The left part of the figure shows a sample logical view of training workers in 3D parallelism. The parallelization strategy is of size MP(4)-DP(3)-PP(2), meaning that there are 4/3/2 peer workers for the MP/DP/PP dimension. Each worker is named with 3 digits, representing the ID of the worker in the MP, DP, and PP dimensions, respectively. Workers that are aligned along each dimension should *communicate* for that respective dimension’s parallelization type. For example, workers 000, 100, 200, and 300 should communicate for MP type (i.e., activation/input gradient sync during forward-pass/back-propagation), while workers 300, 310, 320 should communicate for DP type (weight gradient sync during backpropagation).](../../papers/fred/figure-01.png)

**図 1.** HW-SW Co-Design Stack for Optimizing DNN training communication. This work addresses the three phases highlighted in red. The left part of the figure shows a sample logical view of training workers in 3D parallelism. The parallelization strategy is of size MP(4)-DP(3)-PP(2), meaning that there are 4/3/2 peer workers for the MP/DP/PP dimension. Each worker is named with 3 digits, representing the ID of the worker in the MP, DP, and PP dimensions, respectively. Workers that are aligned along each dimension should *communicate* for that respective dimension’s parallelization type. For example, workers 000, 100, 200, and 300 should communicate for MP type (i.e., activation/input gradient sync during forward-pass/back-propagation), while workers 300, 310, 320 should communicate for DP type (weight gradient sync during backpropagation).

Communication in DNN training depends inherently on the parallelism strategy being employed. Data-Parallel (DP) [Li20c, An20], Model-Parallel (MP) [Lep20, Jia19a], and Pipeline-Parallel (PP) [Hua19, Pip19] are the building blocks of any parallelism strategy. In DP, the DNN model is replicated across NPUs and each NPU works on a different set of training samples (i.e., minibatch). In MP, each DNN layer is sharded across NPUs while they work on the same training samples. In PP, each NPU hosts a subset of DNN layers, and training samples flow through the NPUs in a pipeline manner. 3D parallelism [Eff21] utilizes all the aforementioned strategies by creating different MP/DP/PP groups between NPUs. The optimal balance between DP, MP, and PP is heavily dependent on the workload and underlying platform and can *significantly vary for different workload/platform configurations* [Jia19a, Ali24]. [Figure 1](#figure-01) shows an example of a 3D parallelism strategy.

<span id="figure-02"></span>

![図 2. さまざまな並列化戦略における計算・通信オーバーヘッドの正規化値.](../../papers/fred/figure-02.png)

**図 2.** ウェハ上の 20 個の NPU を接続する 2D Mesh トポロジー (詳細は[第 6 節](#section-6)) で Transformer-17B を実行したときの, さまざまな並列化戦略 (説明は[第 2 節](#section-2)) における計算・通信オーバーヘッドの正規化値.

From a communication perspective, 3D parallelism requires executing *multiple concurrent communication operations* between NPUs within the same MP/DP/PP group at different stages of distributed training. Moreover, different parallelism strategies stress the compute and communication differently. This is quantified in [Figure 2](#figure-02). As the figure shows, high communication overhead can result in the total training overhead of compute-efficient strategies being greater than that of less compute-efficient strategies (e.g., MP(20)-DP(1)-PP(1) vs. MP(5)-DP(4)-PP(1)). Other than the communication volume which is determined by the workload, the main purpose of such high network overhead in [Figure 2](#figure-02) is the *inefficient use of network resources* in the baseline topology. This is mainly because: (i) for the majority of the comm operations, only half or less than half of the NPU links get activated (discussed in detail in [第 3.2.4 節](#section-3-2-4)), (ii) network contention between MP/DP/PP parallel groups (discussed in detail in [第 3.2.2 節](#section-3-2-2)). The full list of baseline topology challenges is discussed in [第 3.2 節](#section-3-2).

In summary, an optimal wafer-scale fabric for distributed DNN training should meet the following three needs:

- Handle *multiple* non-blocking collective communications with minimum congestion.
- Be efficient for *all* 3D parallelism configurations.
- Provide *high-BW* connectivity between NPUs.

In this work, we propose **Fred**, a wafer-scale fabric with Flexible REduction-Distribution feature for supporting arbitrary 3D parallelism. Fred includes: (i) a novel topology with switches that provide native support for reduction and broadcast for bandwidth amplification, (ii) a collective routing algorithm with non-blocking support, and (iii) a device placement algorithm to minimize congestion. We deploy Fred over a wafer-scale substrate [Des21]. Each NPU in our architecture is a hybrid integration of high-end compute chiplets and 3D-stack DRAM chiplets (analogous to H100 [Nvi23]). We also discuss solutions to physically layout and scale the Fred topology over a wafer substrate.

To the best of our knowledge, *Fred is the first wafer-scale fabric proposal tailored for DNN training, that can efficiently support multiple concurrent collectives for hybrid parallelization strategies (e.g., 3D parallelism)*. Hence, Fred enables the compiler to consider any type of parallelization strategy without any concern about how efficiently they can be executed on the network. To summarize:

- We motivate the challenges with designing a wafer-scale fabric for 3D parallelism ([第 3 節](#section-3)).
- We propose Fred, a novel network fabric that includes several innovative features: *a switch fabric* with flexible reduction-distribution trees connected via a scalable topology ([第 4 節](#section-4)), and *a novel routing algorithm* to route multiple collectives concurrently, along with a congestion-aware device placement policy for 3D parallelism ([第 5 節](#section-5)).
- We demonstrate how Fred can be implemented as a wafer-scale fabric ([第 6 節](#section-6)).
- We compare Fred with baseline fabrics for some sample workloads and parallelization strategies ([第 8 節](#section-8)).

Our 結果 show that Fred can improve the average end-to-end training time of ResNet-152, Transformer-17B, GPT-3, and Transformer-1T by $1.76\times$, $1.87\times$, $1.34\times$, and $1.4\times$, respectively, when compared to the baseline 2D Mesh.

<span id="section-2"></span>

## 2 背景

<span id="section-2-1"></span>

### 2.1 集合通信パターン

Although DNN models can be highly diverse, most of their communication during distributed training can be handled through collective patterns [An20]. Depending on the model type and parallelization strategy, different types of collectives may be needed to synchronize on activations/gradients during forward-pass/back-propagation [Ast20]. [Figure 3](#figure-03) shows the mathematical implication of the most common collective patterns between three workers. During *Reduce-Scatter*, workers communicate in such a way that, at the end, each worker has a portion of globally reduced data. In *All-Gather*, each worker broadcasts its local data to all other workers. *All-Reduce* is the most common pattern in distributed training [An20] and can be thought of as a *Reduce-Scatter* followed by an *All-Gather*. In *Reduce*, multiple NPUs participate in reducing data, and the result is stored only on one NPU, while *Gather* collects the data from all NPUs and stores them on a single NPU. *Multicast* means a single NPU sends its data to multiple NPUs. In *All-to-All*, each worker sends a portion of its local data to each worker.

<span id="figure-03"></span>

![図 3. 集合通信パターン among three workers.](../../papers/fred/figure-03.png)

**図 3.** 集合通信パターン among three workers.

<span id="section-2-2"></span>

### 2.2 集合通信アルゴリズム

The patterns described in [第 2.1 節](#section-2-1) can be handled through different *集合通信アルゴリズムs*. In general, there are two distinct way to implement such algorithms:

**1) Endpoint-based.** NPUs communicate in a peer-to-peer distributed manner through explicit send/recv of messages with themselves and without requiring central coordination. In this case, the optimal algorithm is usually dependent on the physical network topology and collective size. For example, ring-based All-Reduce is optimal when the physical topology is a ring, while tree-based All-Reduce is optimal for tree-based topologies, or when the message size is small [Tha05].

One drawback of the NPU-to-NPU-based approach is the amount of traffic it generates. For example, the most BW-optimal NPU-to-NPU algorithms require each NPU to send/receive nearly $\frac{2(N-1)}{N}D$ bytes of data to execute an All-Reduce of $D$ bytes among $N$ NPUs [An20], which is almost $2\times$ of the All-Reduce size ($D$ bytes) [Tha05, Col06, Col07]. This is because all endpoint-based algorithms must perform reduction and gather phases separately, resulting in $\frac{(N-1)}{N}D$ send/recv per NPU to accomplish each phase, respectively [Tha05, Col06, Col07].

**2) ネットワーク内集合通信実行.** To alleviate the extra traffic of endpoint-based approach, recent proposals have introduced in-network 集合通信アルゴリズムs by adding compute capability to the switches [Sca21, An20, Acc19] to perform both reduction and gather at the same time. For example, an All-Reduce of $D$ bytes only requires each NPU to send/receive $D$ bytes to the switch/switch-hierarchy. The switch/switch-hierarchy receives $D$ bytes from each NPU, performs reduction across all received data from all $N$ NPUs, and broadcasts $D$ bytes back to all NPUs. Therefore, compared to the endpoint-based approach, each NPU sends/receives almost half the traffic ($D$ bytes vs. $\frac{2(N-1)}{N}D$ bytes.) [Acc19]. Additionally, ネットワーク内集合通信実行 allows the endpoint resources to be allocated for training compute tasks, while the network switches handle the collectives efficiently.

<span id="section-2-3"></span>

### 2.3 3D 並列化における通信

There are multiple ways of distributing the distributed training tasks across multiple NPUs (a.k.a. the parallelization strategy): MP (a.k.a. Tensor-parallelism) [Sho19], DP [Li20c], and PP [Hua19, Pip19]. The combination of these strategies can be generalized in the form of 3D-parallelism [Eff21]. [Figure 1](#figure-01) shows the concept of 3D-parallelism. In this case, each training worker is part of one MP, DP, and PP group, where the ID (offset) of each NPU within its MP/DP/PP group is determined using the first/second/third digits of a 3-digit worker ID. Therefore, the NPUs that have the same DP & PP digits are within the same MP group (e.g., 000, 100, 200, and 300).

The NPUs within the same DP group should communicate through the *All-Reduce* collective pattern during back-propagation to sync their locally computed model gradients and update the model before starting the next training iteration [An20]. For the MP group case, NPUs need to communicate during forward-pass/back-propagation to synchronize on output-activations/input-gradients. The communication pattern, however, depends on the layer type and the way it is sharded. The usually observed patterns are: *All-Reduce* [Sho19], *All-to-All* [Nau19], *Reduce-Scatter* [Lep20], and *All-Gather* [Lep20]. For the PP group, the NPUs need to transfer the output-activations/input-gradients during forward-pass/back-propagation on the borderline layers and pass the data to the NPU(s) hosting the next set of layers. [Table 1](#table-01) represents collective patterns incurred by each parallelization strategy.

[Figure 1](#figure-01)also shows the necessity to handle multiple collectives at the same time. For e.g., there are eight different DP groups, meaning that up to eight concurrent All-Reduces should be handled for the DP communications (similarly, there are six/twelve concurrent communication operations for MP/PP communications). Moreover, the communication type and peer workers differ across MP, DP, and PP groups. Thus, *it is crucial for the underlying network fabric to be flexible for concurrent and different collective patterns*.

<span id="table-01"></span>

![表 1. Collective patterns incurred by distinct parallelizations.](../../papers/fred/table-01.png)

**表 1.** Collective patterns incurred by distinct parallelizations.

<span id="section-2-4"></span>

### 2.4 マルチチップレット統合

In chiplet-based integration, NPU chips are fabricated and then bonded to a package interconnect (e.g., Si-IF) [Sim19, Des21, Tsm24]. In this approach, components from different technologies (e.g., even DRAM) can be integrated on the package. Additionally, since the chiplets can be tested before integration, this approach has a better yield, supports heterogeneity, and requires less redundancy compared to fully monolithic approaches such as Cerebras [Cer21].

**Multi-chiplet Fabric Topologies.** Recent products and research in multi-chiplet platforms are all based on interconnecting NPUs through a 2D-mesh topology [Arc19, Cer21, Des21, Sim19, Enh24, Chi24a]. Among the main reasons for choosing 2D-mesh for on-package/on-wafer is ease of place & route and area optimality over a 2D substrate [Arc19]. *Thus, in this paper, we choose 2D-mesh as the main baseline topology and compare our proposal against it*.

<span id="section-3"></span>

## 3 ウェハスケールファブリックの目標指標

<span id="section-3-1"></span>

### 3.1 通信要件

First, we discuss two execution modes for running DNN training over a wafer-scale substrate.

<span id="section-3-1-1"></span>

#### 3.1.1 重み定常

When DNN models can fit entirely in the available on-chip memory within a wafer, loading the entire model parameters and collecting the training result to/from the package is a one-time overhead. [+1] The cost of loading the pre-trained and storing the trained model is amortized over thousands of training iterations. The input samples, however, need to be loaded at the beginning of each training iteration. Such I/O operations have minimal impact on the overall training performance since the samples are much smaller than the model size. Therefore, *in this mode, the main performance factor is the efficiency of compute cores and the NPU-to-NPU communication performance*. A non-optimized interconnect can result in poor NPU-to-NPU communication performance for certain parallelization strategies, forcing the compiler to discard some strategies despite their better compute and on-chip memory utilization, solely because of their poor communication performance ([第 3.2 節](#section-3-2)).

<span id="section-3-1-2"></span>

#### 3.1.2 重みストリーミング

When the available on-chip memory is insufficient to fit the model, the execution model shifts to *重みストリーミング* [Cer21, Thi21]. In this scenario, only a subset of DNN layers is loaded onto the package at any given time. After processing these layers, the on-chip storage is reclaimed for the next set of layers. Consequently, the entire model must be loaded onto the chip multiple times during the model training (at least once during the forward pass and once during back-propagation). Additionally, as NPUs compute model gradients, they push this data to off-chip storage, and a lightweight on-storage compute core updates the model for the next iteration [+2] [Cer21]. This approach makes the performance I/O bound, meaning that the upper-bound training performance scales as $\propto\frac{\mathrm{\mathrm{model}}\_\mathrm{\mathrm{size}}}{I/O\_\mathrm{\mathrm{BW}}}$. Therefore, *in addition to compute efficiency and NPU-to-NPU communication performance, maintaining maximum I/O bandwidth is crucial*. A rigid topology can create hotspots when distributing/collecting the model/gradients to/from the I/O channels, which *limits the I/O data rate* ([第 3.2 節](#section-3-2)) and directly impacts training performance.

<span id="section-3-2"></span>

### 3.2 2D Mesh の課題

Next, we discuss specific challenges in a 2D Mesh for supporting the communication needs of DNN training.

<span id="section-3-2-1"></span>

#### 3.2.1 効率的な I/O

As mentioned earlier, maintaining high I/O bandwidth is critical for achieving optimal performance in the 重みストリーミング execution model. However, the 2D mesh often falls short of delivering maximum I/O performance. [Figure 4](#figure-04) illustrates this problem using a $4\times 4$ mesh topology with a pure DP parallelization strategy. In this scenario, each weight fetched from an off-chip memory channel must be broadcast to all NPUs. [Figure 4](#figure-04)(A) shows a broadcast algorithm, based on the *MPI* implementation of one-to-many pattern on 2D mesh [For09], when reading from two different memory channels (shown as the red and blue flows).

Ideally, all memory channels should stream (different) weights simultaneously and with the line-rate to maximize the I/O BW. However, *the shape of 2D Mesh topology inherently generates hotspots and negatively affects the I/O BW*. [Figure 4](#figure-04)(B) shows the maximum channel load, for one hotspot link, when all memory channels are fetching the weights at the same time. If the BW of each memory channel is $P$ bytes/s, then the hotspot link should have the capacity (BW) of $7P$ bytes/s to allow the maximum I/O BW on a $4\times 4$ mesh.

<span id="figure-04"></span>

![図 4. (A) The broadcast communication pattern when reading from two different I/O channels (shown in red and blue arrows). The number associated with each arrow shows the timestamp when data crosses that link for one packet. In practice, multiple packets are pipelined across each path. In this example, the parallelization strategy is MP(1)-DP(16)-PP(1), and the model weights are broadcast among all NPUs for the 重みストリーミング execution model. Note that the reverse order is used to sum the weight gradients during the back-propagation and write the final 結果 into the remote storage. (B) The maximum channel load analysis corresponding to [Figure 4](#figure-04).a, when all of the I/O channels are used simultaneously.](../../papers/fred/figure-04.png)

**図 4.** (A) The broadcast communication pattern when reading from two different I/O channels (shown in red and blue arrows). The number associated with each arrow shows the timestamp when data crosses that link for one packet. In practice, multiple packets are pipelined across each path. In this example, the parallelization strategy is MP(1)-DP(16)-PP(1), and the model weights are broadcast among all NPUs for the 重みストリーミング execution model. Note that the reverse order is used to sum the weight gradients during the back-propagation and write the final 結果 into the remote storage. (B) The maximum channel load analysis corresponding to [Figure 4](#figure-04).a, when all of the I/O channels are used simultaneously.

In general, for an $N\times N$ mesh and $4\times N$ external I/O channels, the wafer-scale fabric links should have a bandwidth of $\mathbf{(2N-1)P}$ bytes/s to fully utilize the I/O bandwidth in all parallelization strategies, assuming each I/O channel has a bandwidth of $P$ bytes/s. As the formula indicates, the required link bandwidth grows $O(N)$ with the mesh width. For larger packages, the technology might not support such high-bandwidth requirements on the package. In such cases, the I/O channel rate must be scaled down proportionally to accommodate the maximum link bandwidth, i.e., $P=\frac{\mathrm{\mathrm{link}}\_\mathrm{\mathrm{BW}}}{(2N-1)}$.

**Fred’s Solution.** Fred prevents network hotspots by adaptively routing the traffic through all of its links equally, enabling further scalability of the wafer-scale systems.

<span id="section-3-2-2"></span>

#### 3.2.2 デバイス配置

<span id="figure-05"></span>

![図 5. Two different device placement mappings for an MP(2)-DP(4)-PP(2) strategy. (A) A device placement that favors MP and DP communications but causes congestion for PP communications. (B) A device placement that favors DP and PP communications but causes congestion for MP communications.](../../papers/fred/figure-05.png)

**図 5.** Two different device placement mappings for an MP(2)-DP(4)-PP(2) strategy. (A) A device placement that favors MP and DP communications but causes congestion for PP communications. (B) A device placement that favors DP and PP communications but causes congestion for MP communications.

Device placement involves assigning each logical training worker to a physical NPU. With $N$ NPUs, there are $N!$ possible device placement mappings. This becomes critical in 3D parallelism, as each training worker may have different communication volumes and patterns with other workers across distinct parallelization groups (refer to [Figure 1](#figure-01)). Therefore, finding a device placement that minimizes network contention is essential.

However, this is challenging with rigid topologies, especially 2D Mesh, where certain communication patterns are inherently prioritized over others. [Figure 5](#figure-05) illustrates two different mappings for a given MP(2)-DP(4)-PP(2) strategy. In [Figure 5](#figure-05)(A), the MP and DP communications are free of congestion, but PP communications cause congestion between different PP groups. Conversely, in [Figure 5](#figure-05)(B), DP and PP communications are optimized, but MP communications face congestion between MP groups. Ultimately, as 2D mesh offers two logically disjoint dimensions ($x$ and $y$), *it is mathematically impossible for all 3D parallelism dimensions to be optimally mapped onto a 2D Mesh*. This is trivial by observing the four corner NPUs, where each NPU offers two outgoing links. Consequently, due to the limited path diversity, one out of the three parallelization groups must experience network congestion and reduced communication performance. Determining which communication patterns to prioritize, unavoidable on 2D Mesh, requires a thorough analysis of the end-to-end workload and understanding the impact of different communication operations.

**Fred’s Solution.** Fred supports congestion-free routing for all communication patterns simultaneously.

<span id="section-3-2-3"></span>

#### 3.2.3 非整列並列化戦略

<span id="figure-06"></span>

![図 6. Network communications on a $4\times 4$ mesh topology for a non-aligned MP(5)-DP(3)-PP(1) parallelization strategy. (A) Non-optimized execution of communication patterns (e.g., All-Reduce) within the NPUs of the same MP group. (B) Traffic congestion between two different DP groups, shown in red and blue, assuming X-Y routing (other group traffic is not shown for clarity).](../../papers/fred/figure-06.png)

**図 6.** Network communications on a $4\times 4$ mesh topology for a non-aligned MP(5)-DP(3)-PP(1) parallelization strategy. (A) Non-optimized execution of communication patterns (e.g., All-Reduce) within the NPUs of the same MP group. (B) Traffic congestion between two different DP groups, shown in red and blue, assuming X-Y routing (other group traffic is not shown for clarity).

When searching for the best parallelization strategy itself, there are many possible configurations where the size of MP/DP/PP is not aligned with the physical topology dimensions. Such configurations create extra challenges on a 2D Mesh, due to the limited path diversity with distinct NPU-to-NPU distances.

[Figure 6](#figure-06)illustrates the communication issues within a $4\times 4$ 2D-mesh topology for an MP(5)-DP(3)-PP(1) strategy. [Figure 6](#figure-06)(A) demonstrates how NPUs in the same MP group need to communicate. Collective communications are often optimized for well-structured topologies (e.g., rings, trees, switches). However, as shown in [Figure 6](#figure-06)(A), the MP groups form non-standard shapes, making it challenging to identify the most optimized 集合通信アルゴリズム for each shape. For example, the distance between NPU 420 and 020 is two hops, due to the rigid shape of 2D Mesh, *making it impossible to construct a well-constructed ring*, even without considering network congestion. [Figure 6](#figure-06)(B) depicts the extra traffic congestion between two different DP groups, marked in red and blue, caused by non-aligned dimensions.

**Fred’s Solution.** Fred provides congestion-free topology and routing mechanisms for any size/placement of MP/DP/PP.

<span id="section-3-2-4"></span>

#### 3.2.4 ネットワーク帯域利用率

Maintaining high bandwidth utilization is challenging for a 2D Mesh. For instance, MP communications are required during both forward-pass and back-propagation phases, while DP communications occur only during back-propagation. However, these links cannot be utilized by MP communications due to the limited paths and lack of optimal routing. Consequently, the links used for DP communication during back-propagation remain underutilized during the forward-pass phase, detrimenting full bandwidth utilization for many strategies on a 2D Mesh.

**Fred’s Solution.** Fred can utilize the full bandwidth of each NPU for every communication phase.

<span id="section-3-2-5"></span>

#### 3.2.5 ネットワーク内集合通信実行

Supporting in-network collectives can significantly reduce network traffic and improve execution performance as described in [第 2.2 節](#section-2-2). This feature, currently employed in off-chip switches [An20, Mel20], requires centralized or hierarchical switches which can perform the collection, reduction, and broadcast of multiple data. A 2D Mesh with distributed NPUs and without a shared central entity, however, impedes the adaptation of the in-network collective support.

**Fred’s Solution.** Fred employs a switch-based topology that supports ネットワーク内集合通信実行.

<span id="section-3-2-6"></span>

#### 3.2.6 まとめ

Ideally, a fabric for DNN training should enable each NPU to fully utilize its network bandwidth for any communication phase of 3D-parallel training without congestion and with support for in-network collectives. These requirements cannot be met via a 2D Mesh, due to their natural shape and rigidity. This underscores the need for the adaptation of new topology and routing mechanisms, such as Fred.

<span id="section-4"></span>

## 4 Fred ネットワークファブリックアーキテクチャ

<span id="figure-07"></span>

![図 7. (a) An overview of the Fred switch with P ports. (b) Fred interconnect (recursively constructed) when the number of ports is even ($2r$) or odd ($2r+1$). (c) Fred<sub>*m*</sub>($2$) switch. (d) Fred<sub>*m*</sub>($3$) switch. (e) R-$\mu$Switch. (f) D-$\mu$Switch. (g) RD-$\mu$Switch. (h) An example of a Fred<sub>*2*</sub>($8$) interconnect implementation and two routed All-Reduce communication patterns (green and orange). (i) Routing Algorithm for three All-Reduce comm flows on Fred<sub>2</sub>($8$) with conflict graph. (j) Example of Routing conflict.](../../papers/fred/figure-07.png)

**図 7.** (a) An overview of the Fred switch with P ports. (b) Fred interconnect (recursively constructed) when the number of ports is even ($2r$) or odd ($2r+1$). (c) Fred<sub>*m*</sub>($2$) switch. (d) Fred<sub>*m*</sub>($3$) switch. (e) R-$\mu$Switch. (f) D-$\mu$Switch. (g) RD-$\mu$Switch. (h) An example of a Fred<sub>*2*</sub>($8$) interconnect implementation and two routed All-Reduce communication patterns (green and orange). (i) Routing Algorithm for three All-Reduce comm flows on Fred<sub>2</sub>($8$) with conflict graph. (j) Example of Routing conflict.

A Fred switch forms the backbone of the fabric. Hierarchical connections of the Fred switches form the full Fred fabric, which is described in [第 6.1 節](#section-6-1). The key idea behind a Fred switch is simple: **break the switch into the most fundamental components, and add small compute capability to each component.** The fine-grained distribution of compute enables supporting flexible and concurrent in-switch collective execution for 3D parallelism communication patterns. In addition, distributed computation of collectives is more scalable to map over the high-BW wafer-scale links than having centralized compute and memory entities.

[Figure 7](#figure-07)(a) shows a Fred switch, which consists of a control unit, input port buffers, and the Fred interconnect. The control unit performs routing between the input ports and the output ports.

The Fred interconnect, shown in [Figure 7](#figure-07)(b), is inspired by *Clos* networks [A53]. Clos networks are identified through the tuple $(m,n,r)$, where $m\geq 2$ is the number of middle stage switches, $n$ is the number of input/output ports per each input/output micro-switch ($\mu$Switch), and $r$ is the number of input/output $\mu$Switches. Fred’s connectivity is similar to the $(m,n=2,r)$ Clos network, which is denoted as Fred<sub>*m*</sub>($P$). $m$ denotes to the number of middle-stage switches, and $P$ identifies the number of input(output) ports. Fred can be designed for an arbitrary number of ports by building on top of the previous works [Arb97]. $P$ is $\frac{2r}{2r+1}$ when $P$ is an $\frac{\mathrm{\mathrm{even}}}{\text{odd}}$ number. Similar to the Clos network, Fred interconnect is constructed recursively, where the middle stage switches are the $\frac{m\times\text{{\mathrm{\mathrm{Fred}}}${}_{m}$($r$)}}{\text{$m\times${\mathrm{\mathrm{Fred}}}${}_{m}$($r+1$)}}$ switches for the $\frac{\mathrm{\mathrm{even}}}{\text{odd}}$ number of ports, as shown in [Figure 7](#figure-07)(b). The recursive design of Fred ends when encountering the base Fred<sub>*m*</sub>($2$) or Fred<sub>*m*</sub>($3$) Switches, which are depicted in [Figure 7](#figure-07)(c) and [Figure 7](#figure-07)(d), respectively.

*The main difference of Fred, compared to a baseline Clos, is adding the reduction and/or distribution (broadcast) support* to the baseline $\mu$Switches. This creates three types of $\mu$Switches depending on which of these two features is present in the $\mu$Switch. [Figure 7](#figure-07)(e) shows the *R-$\mu$Switch* structure that has the reduction feature, i.e., reducing data on the two input ports and routing to one of the output ports. [Figure 7](#figure-07)(f) shows the *D-$\mu$Switch*, which is able to perform distribution by broadcasting one of the input data to both output ports. *RD-$\mu$Switch* is a $2\times 2$ $\mu$Switch and can perform both reduction and distribution, as shown in [Figure 7](#figure-07)(g). The entire Fred switch is built using these three $\mu$Switch types (plus *Muxes* and *Demuxes* to connect the last port to all intermediate stage switches when $P$ is odd) through the recursive process explained earlier.

[Figure 7](#figure-07)(h) shows the complete structure of a Fred<sub>*2*</sub>($8$) switch with two concurrent All-Reduce operations (green and orange). The highlighted $R/D/\mathrm{\mathrm{RD}}$ means that the reduction/distribution/reduction-distribution features of the corresponding $\mu$Switch are activated. For instance, the input $\mu$Switch connecting the input ports $4,5$ performs the reduction and routes the result to one of its output ports. Other non-highlighted $\mu$Switches operate like Clos $\mu$Switches.

<span id="section-5"></span>

## 5 競合のない集合通信ルーティング

<span id="section-5-1"></span>

### 5.1 Fred 上の通信パターン

The fine-grained reduction and broadcast features enable Fred $\mu$Switches to perform all different types of 集合通信パターン observed in distributed training. Collective implementation on Fred, however, can be 概要ed through the notation of *communication flow* (or *flow* in short).

A *flow* on Fred<sub>*m*</sub>($P$) includes a set of input ports ($\mathrm{\mathrm{IPs}}$)={ip<sub>1</sub>, ip<sub>2</sub>, …., ip<sub>*i*</sub>} and output ports ($\mathrm{\mathrm{OPs}}$)={op<sub>1</sub>, op<sub>2</sub>, …., op<sub>*j*</sub>}, where $|\mathrm{\mathrm{IPs}}|\leq P$ and $|\mathrm{\mathrm{OPs}}|\leq P$. The *flow* 結果 in reducing the data across the input ports determined in $\mathrm{\mathrm{IPs}}$ and broadcasting the final result to the output ports identified in $\mathrm{\mathrm{OPs}}$. The port numbers and cardinality of $\mathrm{\mathrm{IPs}}$ and $\mathrm{\mathrm{OPs}}$ can be set independently, depending on the communication pattern. Each communication algorithm can be expressed in terms of performing one or more *flows*. For example, the orange All-Reduce pattern in [Figure 7](#figure-07)(h) is a single *flow* with $\mathrm{\mathrm{IPs}}=\{3,4,5\}$ and $\mathrm{\mathrm{OPs}}=\{3,4,5\}$.

**Simple Communication Algorithms.** Simple communication algorithms refer to communication patterns that can be realized on Fred by performing only one *flow*. [Table 2](#table-02) summarizes different simple Fred 上の通信パターン and the number of involved input/output ports.

**Compound Communication Algorithms.** Compound communication algorithms realize the communication patterns through multiple *flows* on Fred. [Table 2](#table-02) summarizes different compound Fred 上の通信パターン. For example, *Reduce-Scatter* among $i$ inputs is broken into $i$ serial steps of the *reduce* *flow*, and during step $1\leq j\leq i$, the *reduce* operation corresponding to the result of the $op_{j}$ is done. The process is similar for other compound communication algorithms.

<span id="section-5-2"></span>

### 5.2 ルーティングプロトコル

Fred considers a *flow* as a unit of routing, and supports concurrent routing of multiple *flows*. Similar to the previous methods [Nov22], Fred ルーティングプロトコル is also recursive, meaning that first the status of outermost $\mu$Switch levels (i.e., input/output $\mu$Switches) are determined, and then routing is recursively called on the middle stage switches. The difference is, however, supporting reduction/distribution features on the Fred $\mu$switches, and the dependency between the input/output ports of a *flow*, which requires a new routing algorithm to realize these differences. Fred’s ルーティングプロトコル is built upon the following intuitions:

- If two flows share the same input or output $\mu$Switch, they should be routed through different middle-stage switches (subnetworks).
- If both input ports of an R-$\mu$Switch or RD-$\mu$Switch belong to the same *flow*, the reduction feature is activated.
- If both output ports of a D-$\mu$Switch or RD-$\mu$Switch belong to the same *flow*, the distribution (broadcast) feature of the $\mu$Switch is activated.

The latter two points are easy to realize. To satisfy the first point, Fred ルーティングプロトコル creates a *conflict graph*. [Figure 7](#figure-07)(i) shows the first step of a routing example for a Fred<sub>2</sub>($8$) interconnect with the associated conflict graph for this step.

In the conflict graph, each node represents a *flow* and the edges between the nodes represent a conflict (i.e., sharing an input or output $\mu$Switch) between the two nodes (*flows*). Fred routing applies the graph coloring on the conflict graph to find the routing of each *flow*. The number of colors is the number of intermediate stage switches (i.e., $m$). [Figure 7](#figure-07)(i) also shows the 結果 of the graph coloring. Here, there are only two colors since $m=2$. Two flows are routed to the up subnetwork (blue), and one to the down subnetwork (red). After this step, the ルーティングプロトコル and the conflict graph generation are recursively called on the middle blue and red Fred<sub>2</sub>($4$) switches. Note that a desired property of DL training is the deterministic and repetitive nature of its communication patterns that can be inferred at compile time. Therefore, the routing algorithm for different comm phases of the training workload can be executed at compile time and then saved at the control unit of the Fred switches and used during the training to minimize the routing overhead.

<span id="section-5-3"></span>

### 5.3 ルーティング競合と解決方法

There are certain cases where not all *flows* can be routed at the same time, causing *routing conflict*. The routing conflict is identified when the graph coloring fails to color all of the nodes within the conflict graph. [Figure 7](#figure-07)(j) shows an example of a routing conflict when there are four *flows* to be routed on a Fred<sub>2</sub>($8$) and the resulting conflict graph. The conflict graph cannot be colored using only two colors due to the circular dependencies between *flows: 0, 1, 2*. Note that the routing conflict may happen during any recursive call to the routing algorithm (for routing the subnetworks). If the routing conflict is identified, the entire routing is marked to have a conflict.

<span id="table-02"></span>

![表 2. Simple (shaded) and Compound 集合通信アルゴリズムs.](../../papers/fred/table-02.png)

**表 2.** Simple (shaded) and Compound 集合通信アルゴリズムs.

We now discuss ways to address such conflicts.

**(1) Blocking the Conflicting *Flows*.** The first trivial way is to block some of the conflicting *flows* and run them after the other *flows* are finished. This translates to removing some of the nodes in the conflict graph. For example, in [Figure 7](#figure-07)(j), if any of the *flows* $1,2,$ or $3$ is blocked, the routing can proceed to the next step (i.e., subnetworks). This option is, however, costly in terms of performance since it blocks some of the flows.

**(2) Increasing the Number of Middle Stages.** Another method is to design Fred switches with more intermediate stage switches (i.e., increase $m$). This method increases the number of colors for the graph coloring algorithm. Therefore, more conflicting *flows* can be routed simultaneously. [+3] However, this comes at the expense of more HW overhead.

**(3) Decomposing the Communication Algorithms.** For the unicast-only traffic, Fred interconnect is *rearrangeably nonblocking* when $m=2$ and *strict-sense nonblocking* when $m\geq 3$. This fact can be leveraged to decompose some of the communication algorithms into multiple steps and break the dependency among input/output ports in each step (i.e., making them unicast traffic). In the worst case, any 集合通信アルゴリズム can be decomposed into complete unicast traffic. For example, All-Reduce can be handled through a ring-based algorithm at the endpoints (NPUs), rather than in-network execution, which is complete unicast traffic. As a result, *flows* $0,1,$ and $2$ in [Figure 7](#figure-07)(j) can switch to ring-based All-Reduce at the endpoint, while *flow* $3$ uses an in-network All-Reduce algorithm. This method solves the routing by degrading the communication performance of the conflicting *flows* (but it does not block any *flow*).

**(4) Intelligent Device Placement.** Another method to prevent conflicts is through intelligent device placement (mapping) of the training workers to the physical NPUs at the start time. For example, if in [Figure 7](#figure-07)(j) the workers mapped to NPUs of ports $1$ and $4$ swap their locations, the conflict does not happen.

*In Fred, we prioritize the communication performance and do not use options (1) and (3). We use option (2) to simplify the device placement algorithm by only using Fred<sub>*3*</sub>($P$) switches, ensuring that we have three colors in our routing algorithm protocol. Then, for the device placement algorithm, we map the training workers within the same MP group on consecutive physical NPUs, followed by iterating over workers within PP and DP, respectively. This is sufficient to prevent routing conflicts for 3D-Parallelism communication patterns.*

<span id="section-5-4"></span>

### 5.4 重複通信の処理

In training, the workload at a given time may require multiple communication operations. For example, while handling the DP communication in backpropagation, the workload may initiate the PP communication to exchange the next microbatch between the workers. However, FRED’s circuit switch configuration may handle one communication phase at a given time. Additionally, different NPUs might issue communication at different times, due to variations in the compute latencies. Hence, there should be a mechanism to safely preempt the current executing communication operation and execute the new communication, with minimal effects to the in-flight packets, if the latter has a higher priority.

We address this issue by allocating multiple Virtual Circuits (VCs) per port, each dedicated to a specific communication group (e.g., MP), and the FRED’s interconnect to be reconfigured between different overlapping communication operations. While it is possible to frequently reconfigure FRED’s interconnect in short intervals to handle overlapping communication operations concurrently, we choose to reconfigure FRED to execute the highest priority communication operation among the currently pending operations (and preempt the current communication if a new higher priority communication is issued). This decision simplifies the design and minimizes the FRED’s reconfiguration overhead, and is in line with the training workload requirements, since the workload is usually blocking on one communication operation (highest priority) at any given point in time. In our 3D-parallel case, the priority of communication operations in descending order is: MP, PP, and DP. More discussion on FRED’s buffer management and flow control is described in [第 6.2.3 節](#section-6-2-3).

<span id="section-6"></span>

## 6 ウェハスケールアーキテクチャ

We present an instance of a wafer-scale NPU system connected using Fred, for evaluation purposes. We note that alternate configurations are also feasible.

<span id="section-6-1"></span>

### 6.1 Fred ファブリックのレイアウト

A Fred switch builds a foundation to connect multiple wafer-scale NPUs. However, for large wafer-scale systems, due to physical limitations such as wiring, area, etc., it is not feasible to connect all of the NPUs through a single Fred switch. Hence, the *Fred fabric* provides a hierarchical design for the scalable connection of large wafer-scale systems. [Figure 8](#figure-08) shows an example of the Fred fabric that shows a 2-level tree connection of the Fred switches and the NPUs connected to the leaf (*L1*) switches [+4]. In general, tree height and the BW across different levels are determined by the system size and physical 制約 (see [第 6.2 節](#section-6-2)).

When there are multiple levels of Fred switches, the communication algorithms might need to cross several switches and hence, need to be optimized accordingly. For example, [Figure 8](#figure-08)(a) shows the flow path for an All-Reduce between NPUs $1,5,$ and $6$. In this case, the data of NPUs $1\text{ \mathrm{\mathrm{and}} }5$ are reduced on their local L1 switch (to reduce the traffic going to the L2 switch), and the result along with the data of NPU $6$ are reduced on the L2 switch. The final result is sent back to the corresponding L1 switches. The L1 switch attached to NPUs $1\text{ \mathrm{\mathrm{and}} }5$ also multicasts the result to the NPUs.

<span id="figure-08"></span>

![図 8. Physical and Logical Views of 2-level Fred Topologies.](../../papers/fred/figure-08.png)

**図 8.** Physical and Logical Views of 2-level Fred Topologies.

<span id="section-6-2"></span>

### 6.2 ウェハスケールアーキテクチャの構成

We assume a standard 300 $mm$ wafer diameter, similar to the prior works [Arc19, Sca21a], resulting in a 70000 $mm^{2}$ wafer area.

<span id="section-6-2-1"></span>

#### 6.2.1 制約

Fundamentally, there are two physical limitations that limit the amount of compute and other resources on the wafer: (i) Thermal 制約, and (ii) Power delivery network [Des21, Arc19, Sca21a]. Thermal 制約 limit the amount of power that can be delivered to the wafer, depending on the cooling mechanism. Previous works report the maximum power limit within the $9.6\>\mathrm{\mathrm{KW}}$ [Arc19] to $15\>kW$ [Cer21] range. In this paper, we assume $\boldsymbol{15\>kW}$ power is available for the wafer-scale system. The other limitation is the power delivery network, which might necessitate using big on-wafer *voltage regulator modules (VRMs)*, limiting the available area for NPUs [Arc19]. However, alternative solutions can eliminate the need for on-wafer VRMs by either supplying the voltage from the top of the wafer [Cer21], or delivering the power from the back of the wafer by using the *through-wafer-vias (TWVs)* [Pro19]. In this paper, we assume the **on-wafer VRMs are not used** by using any of the solutions described earlier.

<span id="section-6-2-2"></span>

#### 6.2.2 物理システムパラメータ

[Table 3](#table-03) shows the other set of physical parameters. We assume that the NPU chiplets are tested before bonding. If Known Good Die testing is difficult, larger chiplets such as NPU Compute may need to be broken into smaller constituents. Recent work [Chi23c] has suggested that these chiplets actually need to be moderately large (40$mm^{2}$-400$mm^{2}$) in size for cost-optimality. For the purposes of our evaluation, we assume an H100 GPU-like NPU compute chiplet, each equipped with five stacks of HBM3 chiplet memories, resulting in combined power consumption of $700\>W$ and an area of $1314\>mm^{2}$ [Nvi23].

The NPU compute chiplet perimeter can support up to 12 TBps wafer-scale BW, where $6$ TBps of it is allocated to support the 3 TBs local HBM memory BW ($3$ TBps for read + $3$ TBps for write), and the other $6$ TBps is allocated to support $3$ TBps bi-directional total NPU-to-NPU BW ($3$ TBps for send + $3$ TBps for receive).

The $15\>\mathrm{\mathrm{KW}}$ power budget limits the total amount of NPUs on the wafer to $15\>\mathrm{\mathrm{KW}}/700\>W\approx 21$, excluding other component power overheads (e.g., I/O controller, wafer-scale wires). This anticipated power density of 22W/cm<sup>2</sup> is well within the projection of cooling capability in heterogeneous integration roadmaps [Iee23]. In this paper, we consider a $20$-NPU wafer-scale system to make room for other component power overheads. Additionally, $18\times$I/O Controllers are used to connect the wafer to the external memory. Hence, the total NPU $+$ I/O Controller area overhead is $26640\>mm^{2}$.

Similar to [Arc19], we assume in the baseline, the NPU chips are placed with a 100 $um$ distance from each other. Combined with the I/O controllers, the entire baseline can be fit within a rectangle with the size of 190.8 $mm$ $\times$ 150.4 $mm$ in the center of the wafer, leaving the rest of the wafer area unclaimed.

<span id="table-03"></span>

![表 3. 物理システムパラメータ.](../../papers/fred/table-03.png)

**表 3.** 物理システムパラメータ.

<span id="table-04"></span>

![表 4. HW overhead of Fred implementation of [Figure 8](#figure-08)(b).](../../papers/fred/table-04.png)

**表 4.** HW overhead of Fred implementation of [Figure 8](#figure-08)(b).

<span id="section-6-2-3"></span>

#### 6.2.3 Fred トポロジーとパラメータ

To motivate Fred, we leverage the fact that the combination of a constrained power budget and high-end NPUs 結果 in utilizing $26640\>mm^{2}$ out of $70000\>mm^{2}$ area, **making room to utilize otherwise unclaimed area for flexible fabrics like Fred**. However, any fabric proposal must have low power consumption since most of the power budget is allocated to the NPUs. **We demonstrate that Fred meets these properties.**

Our target Fred topology is similar to [Figure 8](#figure-08)(a), where $20$ NPUs and I/O controllers are connected through a 2-level (almost) fat-tree topology. Similar to the baseline, the BW/NPU is still $3$ TBps, but the bisection BW is increased to $30$ TBps. It is almost fat-tree since the L1-to-L2 BW is the summation of attached NPU BW only (and not NPU $+$ I/O Controller). The reason is that if one participant of any *flow* (e.g., *Reduce*) is an I/O controller, then the entire *flow’s* BW requirement is determined by the I/O controller’s BW (e.g., 128 GBps), which is significantly less than NPU-to-NPU BW. Hence, an almost fat-tree gives the same performance as the full fat-tree.

Looking at the BW requirements of Fred L1/L2 switches in [Figure 8](#figure-08)(a), it is clear that each switch chiplet requires a perimeter, to connect the wafer-scale network wires, that is not feasible to build. Hence, in reality, each of the Fred switches in [Figure 8](#figure-08)(a) is decomposed into multiple lower-BW Fred chiplets. [Figure 8](#figure-08)(b) shows a logical view of implementing the (almost) fat-tree based topology of [Figure 8](#figure-08)(a) using feasible Fred chiplets. As [Figure 8](#figure-08)(b) shows, each switch of [Figure 8](#figure-08)(a) is implemented by decomposing it into multiple smaller, but feasible, Fred switches (enclosed in the strip line). For our evaluations, we use Fred<sub>3</sub>($P$) switches.

As [Figure 8](#figure-08)(b) shows, in Fred fabric, L1 switches have hybrid BW downstream links to connect to the NPUs and I/O controllers. This requires Fred L1 switches to use different interface circuitry for NPU vs. I/O controller links, which is accounted for in the overhead numbers in [Table 4](#table-04). In general, hybrid on-chip interconnects are widely used in many designs (e.g., to connect on-chip routers vs. memory controllers in multi-core processors) [Pri04].

**Flow Control.** We assume a Virtual Cut-Through flow control with a credit-based backpressure mechanism to guarantee the switch buffer as packets flow through FRED’s fabric. To enable preemptive communication execution, we consider four VCs per port: three data VCs dedicated to MP, DP, and PP packets and one control VC for the ACK/NACK and other control messages. The data/control packet size is 4KB/512B, with each flit size set to be 512B. The packet header size is 6B to allow for large sequence numbers. Each packet header also has the index to the $\mu$Switch configuration bits, stored in the control unit for a specific communication phase [+5]. If all ports receive a packet belonging to a higher priority phase, Fred changes its $\mu$Switch configuration to that phase and starts forwarding the packets from that phase. Additionally, there is a default header index, which refers to a phase where all flows are unicast and Fred falls back to the online routing to determine the $\mu$Switch configs. While not present in our workloads, this mode is useful when dealing with communication patterns such as *alltoallv* where different src/dst pairs have different size flows that are changing dynamically.

The retransmission protocol is set to be simple Go-Back-N, with an accumulative ack per every 16 data packets to reduce the ack overhead to less than $1\%$ of the network BW. If a switch receives a NACK from an NPU, it forwards it to all input ports participating in that flow, which is then propagated to all NPUs serving as the source of the flow, and retransmission starts from the NACKed packet.

Additionally, each input port has a 24KB buffer per data VC and a 2KB buffer for the control VC. These policies ensure that in the case of communication preemption, there are enough buffers available (i.e., $\mathrm{\mathrm{link}}\_\mathrm{\mathrm{BW}}\times \mathrm{\mathrm{RTT}}=\text{24\mathrm{\mathrm{KB}}}$) for the new communication operation to send at the full link BW.

**HW Overhead.** [Table 4](#table-04) shows the overheads of our proposed Fred implementation shown in [Figure 8](#figure-08). We assume 1.5KB SRAM per FRED switch to store the $\mu$Switch configurations for different communication operations. The numbers are obtained post layout using 15nm NanGate PDK. The total power overhead is $179.35\>W$, which is about $1.2\%$ of the total power budget. The total area overhead is $25195\>mm^{2}$, which can be accommodated by using the unclaimed area available on the wafer. Note that, as discussed in [第 6.2.3 節](#section-6-2-3), the main area overhead of the Fred chiplets is due to I/O for supporting high-BW wafer-scale interconnects, and not because of the switch logic overhead.

**Discussion: Fred Area Overhead.** As we discussed earlier, the unclaimed area on the wafer allows for designing large (but low power) Fred switches to deliver high I/O BW requirements for our topology. In fact, Fred’s internal logic occupies less than 5% of the chip area. Hence, the area overhead of Fred can be significantly reduced if the I/O density increases.

<span id="table-05"></span>

![表 5. Target configurations.](../../papers/fred/table-05.png)

**表 5.** Target configurations.

In our design, we conservatively assume the switch chips use the same interconnect technology as the NPUs (e.g., pitch, frequency, etc.). However, switch area can be further reduced by applying more aggressive network bandwidth technologies. Next generation of I/O technology is expected to deliver up to 250 GBps/mm (compared to 107.4 GBps/mm in our design) [Het24]. This 結果 in designing Fred switch chips with only 18.4% of current area with the same I/O BW.

The other I/O technology alternative is using the serialized high-speed links such as UCIe Advanced [Uni24], which can deliver up to 1 TBps/$mm$. This 結果 in designing Fred switch chips with only 5% of the current area. Note that even with the high area assumption of Fred, we don’t expect the yield issue to be a practical problem since compared to the compute NPUs, Fred switches have much less internal logic and hence encounter fewer defects.

<span id="table-06"></span>

![表 6. Target workloads.](../../papers/fred/table-06.png)

**表 6.** Target workloads.

<span id="section-7"></span>

## 7 評価方法

<span id="section-7-1"></span>

### 7.1 ベースラインと Fred の構成

**Baseline.** The baseline topology is a $5\times 4$ 2D-mesh with I/O controllers attached to the edge NPUs, similar to prior multi-chiplet wafer-scale prototypes [Arc19, Des21, Sim19, Enh24, Chi24a]. Since each NPU has 3 TBps bandwidth ([第 6.2.2 節](#section-6-2-2)), each NPU-to-NPU link in the 2D-Mesh is equal to $750$ GBps, resulting in the bisection BW of 3.75 TBps. The I/O Controller-to-NPU is $128$ GBps.

**Fred.** We test four different variations of Fred to show how different features of Fred contribute to the overall performance. [Table 5](#table-05) shows the target configurations. *Fred-A* shows the effect of going from mesh to switch-based topology with the same bisection and without ネットワーク内集合通信実行. *Fred-B* builds on top of Fred-A and adds the ネットワーク内集合通信実行 feature. *Fred-C* increases the bisection BW without ネットワーク内集合通信実行. Finally, Fred-D is the most optimal variant of Fred by adding the ネットワーク内集合通信実行 to the previous variant.

<span id="section-7-2"></span>

### 7.2 集合通信アルゴリズム

For the baseline 2D mesh and when there is a wafer-wide collective, we use the hierarchical 2D algorithm with two concurrent chunks (in reverse direction) to enhance utilization [Hig20, Enh24]. For collectives between arbitrary NPUs, we build logical rings between involved NPUs and perform the ring algorithm. We also use X-Y routing, which is common in real systems [Hig20]. For Fred-A and Fred-C, we use the hierarchical 2-D ring algorithm to reduce the traffic of L1-L2 links, similar to [Cho19]. Fred-B and Fred-D use the in-network capability and use the hierarchical Fred switch topology to perform the collective, as explained in [第 6.1 節](#section-6-1).

<span id="section-7-3"></span>

### 7.3 対象ワークロードと実行モード

In the interest of space, we evaluate four training workloads, ranging from 60M to 1T parameters to be the representative for a broad range of ML workloads. [Table 6](#table-06) shows the target workloads and their corresponding parallelization strategy and execution models studied in [第 8.2 節](#section-8-2). ResNet-152 and Transformer-17B (Transformer model with 17 billion parameters) can fit on the on-wafer memory and hence, use the *重み定常* execution mode ([第 3.1 節](#section-3-1)). In contrast, GPT-3 and Transformer-1T (Transformer model with 1 trillion parameters) use the *重みストリーミング* execution mode ([第 3.1 節](#section-3-1)). Workers within the same DP group perform All-Reduce together during the back-propagation to sync on weight gradients. In *重み定常* mode, the workers use the Microsoft ZeRO optimizer stage 2 [Raj20b] along the DP dimension to reduce the memory footprint. Note that in *重みストリーミング* mode, the DP groups should reduce the gradients as they stream them out to the external memory through the I/O controller. The pattern is the reverse communication direction of [Figure 4](#figure-04). For Transformer-17B, GPT-3 and Transformer-1T, the model split is based on the Megatron-LM method [Sho19], which requires two All-Reduces (along the MP dimension) for each transformer layer stack during forward-pass & back-propagation. For the PP split on Transformer-17B, we assume the minibatch is divided into 8 microbatches to hide the effect of pipeline bubbles [Hua19]. For GPT-3, however, pipelining works differently since it is combined with the 重みストリーミング. In this case, $\mathrm{\mathrm{PP}}\>=\>2$ indicates that each time $2$ consecutive layers are brought to the wafer and distributed among different NPUs along the PP dimension. Thus, splitting the minibatch into two microbatches is enough to hide the pipeline latency. In [第 8.1 節](#section-8-1) and [第 8.2 節](#section-8-2), the minibatch size for all workloads is set to DP_size$\times 16$, while in [第 8.3 節](#section-8-3) (and also [Figure 2](#figure-02)) the minibatch size is increased to DP_size$\times 40$ to allow for finer-grain pipelining when PP_size increases [+6]. All workloads use FP16 gradient precision.

<span id="section-7-4"></span>

### 7.4 シミュレーションフレームワーク

We use ASTRA-SIM [Ast20, Ast20a], which is an open-source simulation methodology for modeling distributed training systems. ASTRA-SIM enables the profiling of compute and communication performance of distinct wafer-scale fabrics, including Fred. It can model さまざまな並列化戦略 and the overlapping of compute with comm kernels. Additionally, its network back-end can simulate the comm operations in detail. We extend ASTRA-SIM to model the I/O-to-wafer transfers for both the 重み定常 and 重みストリーミング scenarios. For each workload, we run the simulation for two training iterations (i.e., two forward + two backward-pass).

Previous works have shown that endpoint-based collective execution (our baseline) puts more pressure on the endpoint’s compute and memory BW resources, hindering the compute kernel efficiency [Ena21]. To favor the baseline and only focus on the network characteristics, we omit such effects in our baseline system and assume the compute kernels can run as efficient as the ネットワーク内集合通信実行 systems such as Fred.

**Metric of Evaluation.** In [第 8 節](#section-8), we report the end-to-end training times and their breakdowns into total compute time and different *exposed* communication times. Since the minibatch size per training iteration may be different depending on the parallelization strategy, we normalize the reported times by dividing the latencies by the minibatch size when comparing the different parallelization strategies of the same workload (e.g., [Figure 2](#figure-02)). The exposed communication time refers to the amount of time that is not overlapped with the compute time and the workload is waiting for the communication to be finished. Depending on the parallelization strategy and execution model, there might be multiple sources of exposed communication times—load, DP, MP, PP, and/or 重みストリーミング.

<span id="section-8"></span>

## 8 結果

<span id="section-8-1"></span>

### 8.1 マイクロベンチマーク結果

[Figure 9](#figure-09) presents the communication breakdown across 3D parallelism phases for two parallelization strategies for Transformer-17B. For the MP(20)-DP(1)-PP(1) strategy, there are only wafer-wide All-Reduce operations for the MP communication. The baseline effective BW utilization is bounded by the corner NPUs since they have only 2 links to other NPUs. This limits the average ネットワーク帯域利用率 of each NPU to be around $2\times 750\mathrm{\mathrm{GBps}}=1500\mathrm{\mathrm{GBps}}$. In Fred-A, each NPU-L1 BW is 3 TBps, but NPU-L2 BW is 375GBps. [+7] Using a similar analysis as [The22a], we see that hierarchical collectives result in NPU-L2 BW being the bottleneck and the effective NPU BW utilization is $375\mathrm{\mathrm{GBps}}+4\times 375\mathrm{\mathrm{GBps}}=1850\mathrm{\mathrm{GBps}}$. In Fred-B, the L1 switches first perform the All-Reduce and then use the entire L1-L2 BW to forward the data to the L2 switches for the second All-Reduce. Therefore, each NPU can send the data to L2 switch at the speed of $1500\mathrm{\mathrm{GBps}}$ (L1-L2 BW). However, since it is an ネットワーク内集合通信実行, the amount of traffic each NPU sends out is almost half of the traffic in the endpoint-based collective. Fred-C has much more L1-L2 BW and therefore each NPU can drive the BW utilization to $3\mathrm{\mathrm{TBps}}$. In Fred-D, an additional ネットワーク内集合通信実行 reduces the traffic by half in addition to the $3\mathrm{\mathrm{TBps}}$ NPU BW utilization.

<span id="figure-09"></span>

![図 9. Communication microbenchmark 結果 for comparing only communication performance at different phases of 3D-parallelism, for two different parallelization strategies of Transformer-17B from [Figure 2](#figure-02).](../../papers/fred/figure-09.png)

**図 9.** Communication microbenchmark 結果 for comparing only communication performance at different phases of 3D-parallelism, for two different parallelization strategies of Transformer-17B from [Figure 2](#figure-02).

The MP(2)-DP(5)-PP(2) case has all MP (All-Reduce), DP (All-Reduce), and PP (multicast) communications. For the MP communications, the baseline NPU can only utilize 1 link (out of its up to 4 links), resulting in only $750\mathrm{\mathrm{GBps}}$ BW utilization. Since all the communicating NPUs are below the same L1 switch in Fred topologies, they can use the entire $3\mathrm{\mathrm{TBps}}$ of NPU-L1 BW to communicate. Additionally, in the special case when the number of peer NPUs is two, the amount of traffic for endpoint-based vs. in-network execution is the same. Hence, all Fred variants have the same performance for MP communication.

Again, the baseline is limited by the corner NPUs, which can utilize only one of their links for DP communication. Hence, the baseline NPU BW is $750\mathrm{\mathrm{GBps}}$. In Fred, and for the DP communication, each NPU should communicate with four other NPUs under different L1 switches. Therefore, in Fred variants the L1-L2 BW should be shared across four collective flows. Therefore, L1-L2 BW plays a significant role in the performance of this collective. In Fred-A, each NPU has an average NPU-L2 BW of $375\mathrm{\mathrm{GBps}}$, and hence, the NPU BW utilization is only $375\mathrm{\mathrm{GBps}}$, which is worse than the baseline. In Fred-B, however, the L2 switch is used to perform All-Reduce for each flow. This reduces the traffic generated by each NPU roughly by $37.5\%$, which makes its overall performance closer to the baseline. In Fred-C, however, the NPU-L2 BW is increased to $3\mathrm{\mathrm{TBps}}$. Finally, Fred-D Improves the Fred-C by performing in-network collective and reducing the traffic by $37.5\%$.

For the PP comm, the baseline NPU can utilize one of its links to forward data to the next pipeline stage and hence, its BW utilization is 750GBps. Note that this is possible since in the case of language models such as Transformer-17B, one NPU within the mp group is sufficient to multicast the output to all NPUs at the next stage, [+8] and hence, there is no contention between NPUs of the same MP group at the same stage. In Fred, all peer NPUs are below the same L1 switch and can utilize the entire $3{\mathrm{\mathrm{TBps}}}$ BW for the PP comm.

**Discussion: Fred’s NPU to L1 Topology Logic.** Now that we have presented the microbenchmark 結果, we can discuss why we preferred to choose a tree-based topology to connect every four NPUs to the L1 switches. An alternative solution can be a fully-connected topology to connect every four NPUs and then use only one switch level. However, this design choice still suffers from the endpoint-based effects (i.e., increased use of compute and memory BW at the endpoint) discussed in [第 7.4 節](#section-7-4). Furthermore, as explained in [第 2.2 節](#section-2-2), endpoint-based methods produce more communication traffic. For example, in the case of four NPUs, the most endpoint-based BW optimal algorithms produce 1.5D traffic per NPU to perform an All-Reduce of size D [Tha05, The22a], while the ネットワーク内集合通信実行 produces only D traffic per NPU [Sca21], 50% lower than the fully connected topology.

<span id="figure-10"></span>

![図 10. End-to-end training times are decomposed into compute times and different communication times. The runtime of each workload is normalized to its corresponding baseline.](../../papers/fred/figure-10.png)

**図 10.** End-to-end training times are decomposed into compute times and different communication times. The runtime of each workload is normalized to its corresponding baseline.

<span id="section-8-2"></span>

### 8.2 全ワークロード結果: 詳細分析

[Figure 10](#figure-10) shows the end-to-end runtimes of the training workloads for the baseline vs. Fred. Due to space limitations, we only show the Fred-C and Fred-D in comparison with the baseline. However, we note that Fred-A and Fred-B 結果 are between the baseline and Fred-C, in terms of performance. In general, input activations, compared to the model parameters, are relatively small in size and hence, do not have significant overhead on the total iteration time. Additionally, the input activations of the next iteration can be prefetched to the wafer whenever the wafer-scale interconnect is idle. Hence, we observe **no** *initial_input_load* exposed comm for any of our target workloads, except for the Transformer-1T.

ResNet-152 uses pure DP with a 重み定常 model. Hence, the only communication costs that repeat on each training iteration are the input minibatch loading and DP communication. As explained earlier, in wafer-wide All-Reduce collective, the baseline is able to utilize $1.5$ TBps of NPU BW. Fred-C and Fred-D can achieve $3$ TBps NPU BW but Fred-D can further reduce the network traffic by $\approx 2\times$, resulting in a significant reduction of DP exposed comm. Thus, Fred-C and Fred-D can improve the end-to-end training runtime by $1.41\times$ and $1.76\times$, respectively, for ResNet-152.

Transformer-17B uses all dimensions of the 3D-parallelism and therefore, has all DP, MP, and PP communication overheads. The baseline device placement favors MP, but compromises the PP and DP comms, especially due to the non-aligned parallelization strategy dimensions as explained in [第 3 節](#section-3). Another drawback of the baseline is the underutilized links due to the non-overlapping nature of MP/DP/PP comms (see [第 3 節](#section-3)). Fred-C, on the other hand, does not have the problem of underutilized links and 非整列並列化戦略. It also does not require favoring any of DP, MP, or PP over the other strategies. Fred-D can further improve the MP and DP collectives’ performance due to in-switch collective execution capability. As a result, Fred-C and Fred-D can improve the overall end-to-end training performance by $1.75\times$ and $1.87\times$, respectively.

GPT-3 combines 重みストリーミング with 3D-parallelism. Using the analysis of [第 3 節](#section-3), the baseline topology is unable to stream weights with the full line-rate of I/O controllers. The reason is that the hotspot link requires $(2\times 5\>-1)\times 128\text{ \mathrm{\mathrm{GBps}}}\>=\>1152\text{ \mathrm{\mathrm{GBps}}}$, while link capacity is only $750$ GBps. Therefore, the I/O channels should work with $\frac{750}{1152}=0.65\times$ of the line-rate. The MP/PP comm performance of Fred-C and Fred-D is $\approx 4\times$ better than the baseline, due to the underutilized links in the baseline. Note that the reason why Fred-C and Fred-D have the same performance for MP collective comm is because dim(MP)=2. In this special case, as explained earlier, end-to-end and in-switch collective execution have the same amount of networking traffic and hence, have the same performance. In total, Fred-D and Fred-C outperform the baseline by $1.34\times$ in terms of overall training time for GPT-3.

Transformer-1T is another 重みストリーミング workload, but with only DP parallelism. As a result, the 重みストリーミング delay is the only communication overhead in addition to the initial input load. The high-performance compute NPUs and limited off-chip I/O BW puts the 重みストリーミング performance directly on the critical path. This means that the NPUs can work with the line-rate of the weight being streamed, and the main limiting factor is how fast all the weights can be streamed. In this case, both Fred-C and Fred-D can leverage the full I/O BW, while the baseline topology can only work with $0.65\times$ of the total I/O BW as explained earlier. Additionally, since I/O controllers are always being utilized for 重みストリーミング, there is no idle time to prefetch the input minibatch of the next iteration during the current training iteration. Hence, the initial input load cannot be hidden, although its overhead is very negligible. In total, using Fred-C/Fred-D improves the training time by $1.4\times$.

<span id="figure-11"></span>

![図 11. Baseline vs. Fred-D for さまざまな並列化戦略 of Transformer-1T and Transformer-17B](../../papers/fred/figure-11.png)

**図 11.** Baseline vs. Fred-D for さまざまな並列化戦略 of Transformer-1T and Transformer-17B

<span id="section-8-3"></span>

### 8.3 さまざまな並列化戦略

To test the efficiency of Fred for different parallelization strategies, we pick two workloads, Transformer-17B and Transformer-1T, and compare the baseline performance vs. Fred-D in [11(a)](#figure-11) and [11(b)](#figure-11), respectively. The Avg. bars are obtained across all parallelization strategies similar to [Figure 2](#figure-02), however, not all individual parallelization strategies of [Figure 2](#figure-02) are shown in [11(a)](#figure-11) and [11(b)](#figure-11) due to lack of space. As can be observed from both figures, Fred-D can significantly improve communication performance and reduce the total exposed communication in all parallelization strategies.

Such improvements make the most compute-efficient (i.e., least compute time) parallelization strategy also to be the be the best parallelization strategy overall. For example, for Transformer-17B the most compute efficient strategy is MP(20)-DP(1)-PP(1). However, this configuration does not have the lowest overall training time in the baseline system due to its huge exposed communication overheads. Thanks to the benefits of Fred-D in reducing the share of exposed communication overheads, this configuration is now the most optimal compared to other parallelization strategies. This is also true for Transformer-1T, where the most compute-efficient strategy (i.e., MP(5)-DP(1)-PP(4)) is now the most optimal strategy.

Overall, when averaged across all parallelization strategies, Fred-D can improve the exposed communication time by $4.22\times$ and $3.92\times$, resulting in training speedup by $1.63\times$ and $1.44\times$ for Transformer-17B and Transformer-1T, respectively.

**Discussion: going beyond a single wafer.** While the main focus of this paper is on providing flexible on-wafer interconnects to allow for more flexible parallelization strategies, here we discuss the possible scenarios when the model cannot fit on a single wafer. The first method is to pyramidically load and unload parts of the model (i.e., 重みストリーミング) as we considered and evaluated in the paper. However, in some cases more than one wafer is needed for training to reduce the training time. In that case, the optimal inter-wafer topology is an open question. Some methods use reduction trees to accumulate the gradients obtained from different wafers [Cer21]. This method, although efficient for data-parallel strategy across the wafers, is not flexible if we consider other parallelization strategies across wafers. A Fred-like inter-wafer interconnect can be constructed to allow for more flexibility across the wafers. In any case, on-wafer Fred topology can work in tandem with the inter-wafer interconnect to form efficient hierarchical collectives. For example, a global all-reduce can be broken into: i) a special intra-wafer reduce scatter performed by Fred where only the boundary NPUs with access to the I/O maintain the 結果, followed by ii) an All-Reduce facilitated by the inter-wafer interconnect where boundary NPUs reduce the data across different wafers, iii) followed by the final intra-wafer special All-Gather done by Fred where the boundary NPUs broadcast the final result to all NPUs within the same wafer.

**Discussion: going beyond 3D Parallelism.** While the main focus of this paper was on the MP/DP/PP parallelism, recently, more parallelization strategies have been proposed. Examples include Expert-Parallelism (EP) [A23b], Context Parallelism (CP) [Dis25], and more customized and non-homogeneous strategies where the parallelization strategy might change layer by layer [Jia19a]. While not quantitatively studied in this paper, we expect that increasing the parallelization strategy dimensions further increases the network congestion and reduces the effective network BW for each parallelism dimension on the baseline 2D Mesh. This highlights the need to have a flexible network fabric such as Fred.

<span id="section-9"></span>

## 9 関連研究

**Accelerator Fabrics.** Prior works on *flexible* DNN Accelerators [Mae18, Sig20, Fle23, Eye19, Cus17] have explored indirect topologies such as Benes/Fat-tree/Clos for efficiently distributing operands and reducing partial sums. This work leverages this concept to build a topology optimized for collectives.

**In-switch Collectives.** The idea of in-switch collective execution has been proposed in many previous works for different network levels. The *P4* language [P14] allows for offloading application-specific tasks to network switches that support the P4 概要 architecture. [Sca21, Atp21] proposed programming datacenter Ethernet switches for offloading the All-Reduce collective for data-parallel training. iSwitch [Acc19] utilizes FPGA logic within switches to offload the All-Reduce functionality for the distributed training of reinforcement learning. Mellanox SHARP [Mel20] is an Infiniband switch architecture for performing collectives. Klenk *et al.* [An20] propose a method to offload collectives to the scale-up (e.g., NVlink[Eva19]) NPU fabric. Clos topologies have also been explored in prior works [Cus17, Atp21]. A fundamental difference between these works and Fred is that they are proposed for *off-chip* networks, which have significantly less BW compared to on-package networks. In many of these solutions, the internal switch BW should be at least $2\times$ and $P\times$ the link BW to be efficient (i.e., line-rate) for All-Reduce and Reduce between $P$ ports, respectively. This is due to the switch architecture that performs the reductions only after the routing and on the output port. While the difference between off-chip links and on-chip switch architectures allows for provisioning such BW differences, it is not applicable for on-package/on-wafer platforms where the links are on-chip and can have the same BW as the switches. In contrast, Fred performs the reduction operations in multiple steps ($\mu$Switches) during the routing on the Fred interconnect. Hence, the Fred switch works with the same BW as the links and can provide line-rate throughput.

<span id="section-10"></span>

## 10 結論と今後の課題

We propose Fred, a high-BW wafer-scale fabric that is flexible for different configurations of the 3D parallelization strategies of distributed training workloads. Fred is able to support concurrent ネットワーク内集合通信実行 efficiently, enabling the upper-level compiler to further optimize the parallelization strategy for compute and memory utilization. We plan to study Fred for distributed inference as a part of our future work.

Acknowledgements.

[+1]: All model updates over different training iterations happen on-chip.

[+2]: Model updates involve low operational intensity. Hence, performing these updates off-chip prevents wasting I/O bandwidth by avoiding loading optimizer states onto the chip for lightweight operations.

[+3]: For example, Fred<sub>3</sub>($8$) can route all the flows in [Figure 7](#figure-07)(j).

[+4]: We note that Fred layout shown is not tiled. This means that the substrate (where chiplets are bonded) may not be able to use stepper-based lithography. But direct-written maskless lithography is not uncommon for substrate patterning. This was used in a commercial packaging provider ThinkDeca [Ada24a]. Such patterning has no symmetry requirement, albeit it has a lower throughput. Also, note that using maskless lithography increases the substrate manufacturing moderately [Cos10], but substrate manufacturing is a small fraction of the total system cost [Des21].

[+5]: Compound collectives have multiple phases

[+6]: For these 結果, we assume the number of microbatches is 1, 10, 20, 20, 20, 40 for the Transformer-17B with $\mathrm{\mathrm{PP}}$ size of 1, 2, 4, 5, 10, 20, respectively. For Transformer-1T, the number of microbatches is equal to the $\mathrm{\mathrm{PP}}$ size.

[+7]: Assuming the L1-L2 BW is equally shared among all NPUs.

[+8]: All NPUs within the same MP group produce the same output in this case
