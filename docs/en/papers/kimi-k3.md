---
title: 'Kimi K3: Open Frontier Intelligence'
createTime: 2026/07/29 15:00:00
permalink: /en/papers/kimi-k3/
---

> [Kimi Team](https://platform.kimi.com/docs/guide/kimi-k3-quickstart). Kimi K3: Open Frontier Intelligence: Technical Report of Kimi K3. This web reading edition follows the text extracted from the [original PDF](/paper/k3_tech_report.pdf). The PDF remains authoritative for mathematical notation, figures, tables, and the complete reference list. [Model weights](https://huggingface.co/moonshotai/Kimi-K3).

## Abstract

We introduce Kimi K3, a 2.8T parameter Mixture-of-Experts model with 104 billion activated parameters, native vision capabilities, and a 1-million-token context window. Kimi K3 is built on Kimi Delta Attention [Tea25b] and Attention Residuals [Tea26], which improve information flow across sequence length and model depth. Together with Stable LatentMoE, which effectively activates 16 of 896 routed experts per token, and refined training and data recipes, these advances yield an approximately $2.5\times$ improvement in overall scaling efficiency over Kimi K2 [Tea25]. Post-training highlights reinforcement learning across general, agentic, and coding domains and multiple reasoning-effort levels, enabling compositional generalization and robust long-horizon execution. At 2.8T scale, Kimi K3 is supported by infrastructure advances in multiple areas: algorithm-system co-design for KDA, perfectly balanced expert-parallel training with efficient memory management, million-token agentic RL with persistent rollout and sandbox states, and deployment innovations. Extensive evaluations show that Kimi K3 achieves frontier-level performance across long-horizon coding, agentic, knowledge, reasoning, and vision tasks. While its overall performance still trails the most powerful proprietary models, namely Claude Fable 5 and GPT-5.6 Sol, Kimi K3 consistently outperforms other open and proprietary models evaluated in our suite. We release the full Kimi K3 model weights to facilitate future research and accelerate the broader deployment and adoption of frontier intelligence.

![Kimi K3 benchmark results](../../papers/kimi-k3/figure-01.png)

**Figure 1.** Kimi K3 main results.

## 1 Introduction

For much of the development of Large Language Models (LLMs), scaling meant investing more computation before deployment by training larger models on more data [Kap20, Hof22]. The rise of reasoning models has established test-time computation as a second axis of scaling: OpenAI's o-series scales reinforcement learning and test-time reasoning [Ope24, Ope25]; Anthropic's extended-thinking models allocate adaptive thinking budgets and interleave reasoning with tool use [Ant25, Int25]; DeepSeek-R1 [Guo25] and Kimi K1.5 [Sca25] show that large-scale reinforcement learning can elicit sophisticated reasoning behaviors from strong pre-trained models; and Kimi K2.5 Agent Swarm [Kim26a] further extends test-time scaling from sequential reasoning to parallel agent coordination. These advances have made test-time scaling a central focus of frontier research. However, while the open-source model ecosystem has advanced rapidly on the second axis, it has progressed slowly on the first: many recent models remain within or slightly above the 1T-class parameter regime [Zen26, Dee26, XiaWeb, Thi26]. As increasingly sophisticated reasoning and agentic reinforcement learning methods are applied to pre-trained foundations of similar scale, open-source progress risks converging while the gap to the strongest proprietary systems widens. With Kimi K3, we pursue both scaling axes together to the frontier: scaling the pre-trained foundation to unprecedented 3T-class parameters while scaling reinforcement learning, reasoning effort, and long-horizon interaction at 1M context length.

We introduce Kimi K3, a native multimodal Mixture-of-Experts model with 2.8 trillion total parameters, 104 billion activated parameters, and a context window of up to one million tokens. Its architecture scales information flow across sequence length, network depth, and model width. Kimi Delta Attention (KDA) [Tea25b] provides efficient long-sequence mixing, with periodically interleaved Gated MLA layers preserving global interaction. Attention Residuals (AttnRes) [Tea26] allows each layer to selectively attend to representations from all preceding layers. Stable LatentMoE expands the routed expert space to 896 experts, with 16 activated per token, while normalization, SiTU-GLU, and Quantile Balancing stabilize optimization at extreme sparsity. These architectural advances, combined with refined data and training recipes, yield an approximately $2.5\times$ improvement in overall scaling efficiency over Kimi K2 [Tea25].

We pair this pre-training foundation with post-training designed explicitly for 1M-context test-time scaling. Kimi K3 undergoes reinforcement learning across long-horizon coding, general agents, general reasoning, and knowledge tasks, each spanning multiple reasoning-effort levels. Training environments include verifiable search and professional knowledge work, software engineering and kernel optimization, multimodal reasoning with vision-in-the-loop tool use, persistent assistant workflows, web development, and autonomous execution tasks. These environments train a general loop of reasoning, acting, observing, verifying, and adapting, often over hundreds or thousands of tool calls and millions of accumulated context tokens. Domain- and effort-specialized policies are consolidated into a unified model through multi-teacher on-policy distillation [Lu25, Xia26, Dee26].

Realizing this regime requires infrastructure that scales with architecture complexity, model size, and trajectory length. For systems co-design for KDA, we develop fused kernels, KDA Context Parallelism, and state-aware prefix caching to make KDA efficient within devices, across devices, and across requests. For 2.8T-parameter MoE pre-training, MoonEP provides perfectly balanced expert execution with static computation shapes and zero-copy communication, while memory-efficient training and multimodal encoder optimizations sustain utilization within bounded memory. For million-token agentic RL, our co-located system combines partial rollouts, external KV-cache retention, adaptive throttling, and resumable microVM sandboxes to preserve long-lived model and environment state. Finally, specialized kernels and cache- and budget-aware fleet scheduling translate these innovations into predictable production serving.

The resulting model establishes a new open frontier. On benchmarks spanning long-horizon coding, agentic, knowledge, reasoning, and vision tasks, Kimi K3 trails the strongest proprietary systems overall--Claude Fable 5 and GPT-5.6 Sol--and is consistently ahead of the other open and proprietary models evaluated in the report.

**Contributions.** Kimi K3 combines a 2.8T-parameter native multimodal MoE model with 104B activated parameters and a 1M-token context window; reinforcement learning across domains and reasoning-effort levels; infrastructure for multi-trillion-parameter pre-training and million-token agentic trajectories; and the release of the full model weights.

## 2 Model Architecture

The Kimi K3 architecture is designed to scale information flow along three complementary dimensions: sequence length, network depth, and model width. Along the sequence dimension, Hybrid Attention combines three Kimi Delta Attention (KDA) [Tea25b] layers with one Gated MLA layer in each block, providing an efficient mechanism for long-context token mixing while retaining selective high-capacity attention ([§2.1](#_2-1-hybrid-attention)). Along the depth dimension, Attention Residuals (AttnRes) [Tea26] enable each module to selectively retrieve representations from the embedding, the current block, and preceding blocks, extending information access beyond conventional sequential residual accumulation ([§2.2](#_2-2-attention-residuals)). Along the width dimension, each attention layer is followed by a Stable LatentMoE layer that performs sparse channel mixing, effectively activating 16 of 896 routed experts for each token ([§2.3](#_2-3-stable-latentmoe)). For native vision, MoonViT-V2 encodes images and videos, and a lightweight projector maps the resulting visual features into the shared embedding space before backbone processing ([§2.4](#_2-4-native-vision)). Together with Per-Head Muon ([§2.5](#_2-5-per-head-muon)), these components provide a unified architecture for scaling information flow across tokens, layers, and channels. Combined with refined training and data recipes, they yield an approximately $2.5\times$ improvement in overall scaling efficiency over Kimi K2. Figure 2 provides an overview of the architecture.

![Kimi K3 architecture](../../papers/kimi-k3/figure-02.png)

**Figure 2.** The Kimi K3 architecture, organized around token, channel, and layer mixing, with a native vision pathway at the input. Each block contains three Kimi Delta Attention (KDA) layers followed by one Gated MLA layer, with each attention layer paired with a Stable LatentMoE feed-forward network. Attention Residuals (AttnRes) use learned pseudo-queries to derive attention weights over the embedding and preceding block outputs, enabling selective information flow across depth.

### 2.1 Hybrid Attention

Kimi K3 uses a layerwise hybrid of linear and global attention, combining KDA [Tea25b] with Gated MLA. Each block contains 3 KDA layers followed by 1 Gated MLA layer, giving a 3:1 mixing ratio. This pattern is repeated throughout the backbone. The two attention mechanisms are described separately below. An additional Gated MLA layer is placed at the end of the backbone, ensuring that the final layer always performs global attention.

#### 2.1.1 Kimi Delta Attention

KDA extends the delta-rule recurrence [Sch21, Yan25] with a channel-wise forget gate [Tea25b]. Consider a sequence of hidden states $\mathbf x_t\in\mathbb R^d$, where $t$ indexes the token position and $d$ is the model hidden dimension. For clarity, we first describe a single attention head, with query and key vectors $\mathbf q_t,\mathbf k_t\in\mathbb R^{d_k}$, value vector $\mathbf v_t\in\mathbb R^{d_v}$, and recurrent state $\mathbf S_t\in\mathbb R^{d_k\times d_v}$. KDA applies channel-wise decay before the delta-rule update:

$$
\mathbf S_t=(\mathbf I-\beta_t\mathbf k_t\mathbf k_t^\top)
\operatorname{Diag}(\boldsymbol\alpha_t)\mathbf S_{t-1}
+\beta_t\mathbf k_t\mathbf v_t^\top,
\qquad
\widetilde{\mathbf o}_t=\mathbf S_t^\top\mathbf q_t.
\tag{1}
$$

Here, $\boldsymbol\alpha_t\in(0,1)^{d_k}$ is the channel-wise one-step retention factor, and $\beta_t\in(0,1)$ controls the delta-rule write strength.

Following Kimi Linear [Tea25b], KDA parameterizes the per-head quantities as

$$
\begin{aligned}
\mathbf q_t^h,\mathbf k_t^h
  &=\operatorname{L_2Norm}\!\left(\operatorname{Swish}\!\left(\operatorname{ShortConv}(\mathbf W_{q/k}^h\mathbf x_t)\right)\right)\in\mathbb R^{d_k},\\
\mathbf v_t^h
  &=\operatorname{Swish}\!\left(\operatorname{ShortConv}(\mathbf W_v^h\mathbf x_t)\right)\in\mathbb R^{d_v},\\
\beta_t^h&=\operatorname{Sigmoid}(\mathbf W_\beta^h\mathbf x_t)\in(0,1),\\
\mathbf z_t^h&=\mathbf W_\alpha^\uparrow\mathbf W_\alpha^\downarrow\mathbf x_t+\mathbf b_\alpha^h\in\mathbb R^{d_k}.
\end{aligned}
\tag{2}
$$

The query, key, and value projections apply ShortConv followed by Swish [Yan25], and the query and key are further normalized with $L_2$ normalization [Yan24b]. The low-rank projection and head-specific bias $\mathbf b_\alpha^h\in\mathbb R^{d_k}$ produce a fine-grained decay logit $\mathbf z_t^h$ for each key channel. The lower-bounded mapping from $\mathbf z_t^h$ to $\boldsymbol\alpha_t^h$ is introduced after the chunkwise formulation below.

**Chunkwise parallel form.** Following Kimi Linear [Tea25b], KDA is recurrent across chunks and parallel within each chunk. For a chunk size $C$, $\mathbf X_{[t]}$ stacks the token vectors in the $t$-th chunk for $\mathbf X\in\{\mathbf Q,\mathbf K,\mathbf V,\mathbf O,\mathbf U,\mathbf W\}$. The matrix $\mathbf S_{[t]}\in\mathbb R^{d_k\times d_v}$ denotes the recurrent state entering chunk $t$. For positions $1\le i\le j\le C$, define the channel-wise cumulative decay

$$
\boldsymbol\gamma_{[t]}^{i\to j}:=\prod_{r=i}^{j}\boldsymbol\alpha_{[t]}^r,
\qquad
\boldsymbol\gamma_{[t]}^r:=\boldsymbol\gamma_{[t]}^{1\to r}.
\tag{3}
$$

As in Kimi Linear, $\boldsymbol\Gamma_{[t]}^{1\to C}\in\mathbb R^{C\times d_k}$ stacks $\boldsymbol\gamma_{[t]}^1,\ldots,\boldsymbol\gamma_{[t]}^C$ row-wise. The UT transform produces $\mathbf U_{[t]}$ and $\mathbf W_{[t]}$, from which we define the pseudo-value term $\widetilde{\mathbf V}_{[t]}:=\mathbf U_{[t]}-\mathbf W_{[t]}\mathbf S_{[t]}$. Given the incoming state $\mathbf S_{[t]}$, all outputs in chunk $t$ are computed in parallel as

$$
\begin{aligned}
\mathbf A_{[t]}&=\operatorname{Tril}\!\left[
(\mathbf Q_{[t]}\odot\boldsymbol\Gamma_{[t]}^{1\to C})
(\mathbf K_{[t]}/\boldsymbol\Gamma_{[t]}^{1\to C})^\top\right],\\
\mathbf O_{[t]}&=
\underbrace{(\boldsymbol\Gamma_{[t]}^{1\to C}\odot\mathbf Q_{[t]})\mathbf S_{[t]}}_{\text{inter-chunk}}
+\underbrace{\mathbf A_{[t]}\widetilde{\mathbf V}_{[t]}}_{\text{intra-chunk}}.
\end{aligned}
\tag{4}
$$

For a matrix $\mathbf M$, $\operatorname{Tril}(\mathbf M)$ sets all strictly upper-triangular entries to zero and retains the lower-triangular entries, including the diagonal. This mask enforces causal interactions within the chunk, and the diagonal is retained because each output reads the state after the current-token update. The first term in $\mathbf O_{[t]}$ carries information from preceding chunks, whereas the second term accounts for interactions within the current chunk. We refer readers to Kimi Linear [Tea25b] for the UT transform and the full derivation of the chunkwise form.

**Lower-bounded decay.** Equation 4 rescales the keys in each chunk by the reciprocal cumulative decay $1/\boldsymbol\Gamma_{[t]}^{1\to C}$. Because $\boldsymbol\Gamma_{[t]}^{1\to C}$ is a product of retention factors in $(0,1)$, this reciprocal can grow without bound and overflow in finite precision [Yan24a, Tea25b]. Kimi Linear controls this numerical range by computing relative decay in log space and dividing each chunk into secondary 16-token tiles [Yan24a, Tea25b]. The off-diagonal tiles can then be computed with dense matrix multiplications on Tensor Cores directly. The diagonal tiles, in contrast, still require explicit position-pair computations, which remain the main intra-chunk bottleneck. Kimi K3 addresses this bottleneck by changing the mapping from the decay logits $\mathbf z_t^h$ to the per-step log-decay $\mathbf g_t^h$. Following GDN and Mamba-2, Kimi Linear uses the negative-Softplus mapping $\mathbf g_t^h=-e^{A_h}\operatorname{Softplus}(\mathbf z_t^h)\in(-\infty,0)^{d_k}$ [Yan25, Dao24, Tea25b]. Kimi K3 instead uses a scaled sigmoid to bound the log-decay from below:

$$
\mathbf g_t^h=g_{\min}\operatorname{Sigmoid}(e^{A_h}\mathbf z_t^h)\in(g_{\min},0)^{d_k},
\qquad
\boldsymbol\alpha_t^h=\exp(\mathbf g_t^h)\in(e^{g_{\min}},1)^{d_k}.
\tag{5}
$$

where $A_h$ is a learnable per-head log-scale and $g_{\min}=-5$ is fixed. We initialize $A_h=0$, and each bias $\mathbf b_\alpha^h$ is initialized following [Tea25b, Dao24, Yan25]. With $g_{\min}=-5$, every retention factor satisfies $\alpha_{t,j}^h>e^{-5}\approx6.7\times10^{-3}$, and the cumulative log-decay over a 16-token tile lies in $(-80,0)$. The corresponding reciprocal rescaling factor is therefore smaller than $e^{80}$ and remains within the BF16 dynamic range. This finite range allows both diagonal and off-diagonal tiles to use dense Tensor Core matrix multiplications, eliminating the position-pair diagonal path. This parameterization is closely related to the lower-bounded recurrence gates in prior work [Qin24a, De24, Pen25]. Figure 3 illustrates the change in decay parameterization and its computational consequence.

![Lower-bounded decay and chunkwise KDA computation](../../papers/kimi-k3/figure-03.png)

**Figure 3.** Lower-bounded decay and its effect on chunkwise KDA computation. Kimi Linear uses an unbounded negative-Softplus mapping, whereas Kimi K3 bounds the log-decay with a scaled sigmoid, allowing all causal tiles to use dense Tensor Core matrix multiplications.

**Full-rank gate.** Finally, Kimi K3 changes KDA's output gate from the low-rank parameterization used by Kimi Linear [Tea25b] to an input-dependent full-rank projection. After applying head-wise RMSNorm [Zha19] to the recurrent output, KDA applies data-dependent output gating [Qiu25]:

$$
\mathbf y_t=\mathbf W_o\!\left[\operatorname{Sigmoid}(\mathbf W_g\mathbf x_t)\odot
\operatorname{RMSNorm}(\widetilde{\mathbf o}_t)\right].
\tag{6}
$$

#### 2.1.2 Gated MLA

Multi-head Latent Attention (MLA), introduced in DeepSeek-V2 [Dee24], compresses the key-value representation of each token into a low-dimensional latent vector $\mathbf c_t=\mathbf W_c\mathbf x_t$. Instead of caching full head-specific keys and values, MLA caches $\mathbf c_t$ and reconstructs the content keys and values through learned up-projections during attention computation. This factorization reduces the KV-cache footprint while retaining global token-to-token attention. MLA was subsequently adopted by Kimi K2 and Kimi K2.5 [Tea25, Kim26a], and Kimi K3 retains it in the periodic global-attention layers.

Unlike Kimi K2 and Kimi K2.5, Kimi K3 follows the hybrid design of Kimi Linear [Tea25b] and applies No Position Encoding (NoPE) to all MLA layers. Consequently, no explicit positional encoding is applied to their queries or keys. The intervening KDA layers provide position-sensitive and recency-aware sequence mixing, while the MLA layers provide unrestricted global content interaction. This separation also avoids modifying positional-encoding parameters when extending the context length, such as retuning a RoPE frequency base or applying YaRN [Pen23].

In addition, Kimi K3 augments MLA with an input-dependent, channel-wise full-rank output gate. Let $\widetilde{\mathbf o}_t$ denote the ungated MLA output at position $t$; the gated output is

$$
\mathbf y_t=\mathbf W_o\!\left[\operatorname{Sigmoid}(\mathbf W_g\mathbf x_t)\odot\widetilde{\mathbf o}_t\right].
\tag{7}
$$

The gate projection $\mathbf W_g$ is full rank, matching the new parameterization used by KDA in Kimi K3. This gate allows each token to modulate the channels read from global attention [Qiu25].

To correct the biased rounding error that arises in flash attention, we adopt the method of [Qiu26] and keep the attention output in FP32 during training. This choice doubles the on-chip footprint of the output tile; we therefore redesign the training kernel to overlap it with the KV staging buffers instead of the query tile, freeing shared memory for a deeper KV pipeline and higher training throughput.

### 2.2 Attention Residuals

Standard residual connections [He16] compress all prior information into a single state $\mathbf h_l$ over depth, a bottleneck reminiscent of RNNs over time. For sequence modeling, the Transformer replaced recurrence with attention [Bah14, Vas17], allowing each position to selectively access all previous positions with data-dependent weights. Attention Residuals (AttnRes) [Tea26] applies the same methodology to depth: each layer selectively retrieves representations from all preceding layers rather than accumulating them uniformly.

#### Full Attention Residuals

For each layer $l$, we define a layer-specific learnable pseudo-query $\mathbf q_l=\mathbf w_l\in\mathbb R^d$ and keys and values

$$
\mathbf k_i=\mathbf v_i=
\begin{cases}
\mathbf h_1,&i=0,\\
f_i(\mathbf h_i),&1\le i\le l-1.
\end{cases}
\tag{8}
$$

where $f_i(\mathbf h_i)$ is the output of layer $i$ and $\mathbf h_1$ is the token embedding. The attention weights follow a softmax kernel $\phi(\mathbf q,\mathbf k)=\exp(\mathbf q^\top\operatorname{RMSNorm}(\mathbf k))$ [Kat20, Zha19], where RMSNorm prevents layers with large-magnitude outputs from dominating the weights:

$$
\alpha_{i\to l}=\frac{\phi(\mathbf q_l,\mathbf k_i)}
{\sum_{j=0}^{l-1}\phi(\mathbf q_l,\mathbf k_j)},
\qquad
\mathbf h_l=\sum_{i=0}^{l-1}\alpha_{i\to l}\mathbf v_i.
\tag{9}
$$

Since network depth is modest ($L<100$), the $O(L^2d)$ arithmetic of this full form is affordable; the practical overhead is the $O(Ld)$ memory and cross-stage communication under pipeline parallelism required to keep all layer outputs alive.

#### Block Attention Residuals

To reduce this overhead, we partition the $L$ layers into $N$ blocks of $S=L/N$ layers each. Within block $n$ (layer indices $\mathcal B_n$), layer outputs are reduced to a single representation by summation, $\mathbf b_n=\sum_{j\in\mathcal B_n}f_j(\mathbf h_j)$, with $\mathbf b_n^i$ denoting the partial sum over the first $i$ layers of the block; we set $\mathbf b_0=\mathbf h_1$ so the token embedding is always included as a source. Across blocks, full attention is applied over only the $N$ block-level representations: for the $i$-th layer in block $n$, the value matrix is

$$
\mathbf V=
\begin{cases}
[\mathbf b_0,\mathbf b_1,\ldots,\mathbf b_{n-1}]^\top,&i=1,\\
[\mathbf b_0,\mathbf b_1,\ldots,\mathbf b_{n-1},\mathbf b_n^{i-1}]^\top,&i\ge2.
\end{cases}
\tag{10}
$$

with keys and attention weights following Equations 8 and 9. The final output layer then aggregates all $N$ block representations. Under Block AttnRes, memory and communication overhead drop from $O(Ld)$ to $O(Nd)$, while this block structure also bounds the inference-time state, enabling the parallel inter-block results to be better merged with the sequential intra-block partial sums via online softmax [Mil18], significantly reducing inference time cost.

Empirically, $N\approx8$ recovers most of the benefit across model scales [Tea26]; for Kimi K3, we partition its layers into 8 blocks with a 12-layer block size, giving a partial final block and 9 total blocks when counting the embedding layer.

### 2.3 Stable LatentMoE

Increasing both the expert pool and the number of active experts expands the space of expert specializations, but in a conventional MoE each selected expert receives the full $d$-dimensional token representation, so communication and expert-weight traffic grow with the routing multiplicity. LatentMoE [Fed22] makes this expansion affordable by separating the full model width from the routed-expert width: shared experts retain a full-width path for common transformations, whereas specialized routed experts operate in a compact latent space of width $\ell$. This enables Kimi K3 to scale channel mixing to 896 routed experts with 16 active experts per token, corresponding to a sparsity of 56.

This extreme sparsity amplifies two failure modes of the vanilla design. First, the routed path composes $\mathbf W^\downarrow$, a gated multi-branch expert feed-forward network, and $\mathbf W^\uparrow$ into a chain of nearly four consecutive matrix multiplications. This ill-conditioned structure, combined with the 2.8-trillion-parameter scale, produces exploding internal activations in the routed branch. Second, balancing the load of nearly $10^3$ experts exceeds the regime in which existing auxiliary-loss-free bias updates remain well behaved. Stable LatentMoE addresses these two failure modes with three components: an RMSNorm before the up-projection and Sigmoid Tanh Unit GLU (SiTU-GLU) to suppress activation explosion, and Quantile Balancing (QB) for load balancing.

As illustrated in Figure 2, the layer follows the shared- and routed-expert organization of DeepSeekMoE [Dai24]. For $\mathbf x\in\mathbb R^d$, the shared experts process $\mathbf x$ directly, while the routed path projects it to $\mathbf z=\mathbf W^\downarrow\mathbf x\in\mathbb R^\ell$, dispatches $\mathbf z$ to the selected experts, and maps their weighted aggregate back to $\mathbb R^d$ through $\mathbf W^\uparrow$:

$$
\mathbf u=\sum_{i\in\mathcal T_k(\mathbf x)}p_iE_i^{\mathrm{routed}}(\mathbf W^\downarrow\mathbf x),
\qquad
\mathbf y=\sum_{j=1}^{N_s}E_j^{\mathrm{shared}}(\mathbf x)
+\mathbf W^\uparrow\operatorname{RMSNorm}(\mathbf u).
\tag{11}
$$

Here, $\mathbf u\in\mathbb R^\ell$ is the aggregated routed representation, $E_j^{\mathrm{shared}}:\mathbb R^d\to\mathbb R^d$ and $E_i^{\mathrm{routed}}:\mathbb R^\ell\to\mathbb R^\ell$ are the shared and routed expert feed-forward networks, and $p_i$ is the router weight defined by the Quantile Balancing rule below. Kimi K3 fixes the number of full-width shared experts to $N_s=2$ in every layer.

#### 2.3.1 Normalized LatentMoE

The original LatentMoE directly applies $\mathbf W^\uparrow$ to the aggregated routed representation $\mathbf u$, whose scale can vary with the selected experts and their routing weights. As shown in Equation 11, Kimi K3 instead inserts RMSNorm [Zha19] between expert aggregation and the up-projection. This normalization reduces the sensitivity of the routed branch to scale variation before it is combined with the full-width shared branch. Beyond stabilizing training, the additional RMSNorm consistently improves validation loss and downstream benchmarks.

#### 2.3.2 Sigmoid Tanh Unit GLU

Gated Linear Units (GLUs) modulate a linear value branch with a sigmoid-activated gate, computing $\operatorname{Sigmoid}(\mathbf W_g\mathbf x)\odot\mathbf W_u\mathbf x$ [Yan17]. SwiGLU replaces the sigmoid gate with $\operatorname{Swish}(x)=x\operatorname{Sigmoid}(x)$ and yields strong empirical performance in Transformers [Sha20]. SwiGLU has subsequently become a widely adopted FFN design in large language models, while a complete account of its empirical effectiveness remains open. However, both multiplicative factors in SwiGLU are unbounded, so coincident large coordinates can produce activation outliers and increase overflow risk in low-precision arithmetic. The sigmoid gate of the original GLU avoids unbounded gate growth, but it does not retain the approximately linear positive regime of Swish. This motivates an activation that controls large-value growth while preserving the characteristic local and positive-side response of SwiGLU. Other recent efforts have explored alternative parameterizations of this trade-off [Jia26].

To satisfy these requirements, we propose Sigmoid Tanh Unit GLU (SiTU-GLU). SiTU-GLU applies the smooth cap $\operatorname{softcap}(x,\beta)=\beta\tanh(x/\beta)$ to the linear factor of the Swish gate and independently to the up branch:

$$
\operatorname{SiTU\text{-}GLU}(\mathbf x)=
\beta_1\tanh(\mathbf W_g\mathbf x/\beta_1)
\odot\operatorname{Sigmoid}(\mathbf W_g\mathbf x)
\odot\beta_2\tanh(\mathbf W_u\mathbf x/\beta_2).
\tag{12}
$$

For Kimi K3, we set the soft-cap hyperparameters to $\beta_1=4$ for the gate branch and $\beta_2=25$ for the up branch. The scaled tanh is approximately linear near the origin and bounded at large magnitude, allowing SiTU-GLU to preserve the local response of SwiGLU while controlling both factors in the product. Figure 4 compares the branch definitions and scalar responses of GLU, SwiGLU, and SiTU-GLU on a common slice. Appendix B gives the local expansion, limiting case, formal output bound, and comparison with hard clamping.

![GLU, SwiGLU, and SiTU-GLU branch responses](../../papers/kimi-k3/figure-04.png)

**Figure 4.** Gate and up branches of GLU, SwiGLU, and SiTU-GLU, together with their scalar responses. SiTU-GLU closely follows SwiGLU near the origin and approaches the bound $|f(x)|\le\beta_1\beta_2=100$ for large positive inputs, whereas SwiGLU remains unbounded.

#### 2.3.3 Quantile Balancing

Unlike auxiliary-loss-based routing [Fen25], Kimi K3 adopts auxiliary-loss-free routing [Dee24a]. Load balancing is implemented by adding an expert-specific bias $b_j$ to the router score used for Top-$k$ selection. For token $\mathbf x_i$, the router computes $\mathbf s_i=\operatorname{Sigmoid}(\mathbf W_r\mathbf x_i)$ and applies

$$
\mathcal T_i=\operatorname{argtop}_k(\mathbf s_i+\mathbf b),
\qquad
p_{i,j}=\frac{s_{i,j}}{\sum_{r\in\mathcal T_i}s_{i,r}},\quad j\in\mathcal T_i.
\tag{13}
$$

Because $\mathbf b$ is omitted from $p_{i,j}$, it regulates dispatch without altering the mixture weights or gradient-based router optimization. The original method updates $\mathbf b$ with the fixed-step rule [Dee24a]

$$
b_j^{(t+1)}=b_j^{(t)}+\gamma\operatorname{sign}(\ell-\ell_j^{(t)}),
$$

for which $\gamma$ trades off slow adaptation against load oscillation. Maintaining balanced loads becomes more challenging as LatentMoE increases the routed expert pool to 896 per layer. Imbalanced routing slows expert-parallel training and may leave some experts poorly trained [Hua26].

To address this limitation, we introduce Quantile Balancing (QB), which sets each expert bias from the router-score quantile that matches its target load [Su26]. Consider a training batch of $m$ tokens routed to $n$ experts with Top-$k$ selection, so the target load is $q:=mk/n$ tokens per expert. QB derives the next bias from a single forward pass. Routing replaces Top-$k$ with Top-$(k+1)$ on the biased score $\mathbf s_i+\mathbf b^{(t)}$: the first $k$ entries are the routes actually taken, while the $(k+1)$-th entry is the cutoff $\alpha_i^{(t)}$ that an expert must exceed to enter token $i$'s Top-$k$.

Taking the cutoff from Top-$(k+1)$ routing avoids a separate token-side quantile. With the cutoffs fixed, the token count routed to expert $j$ under a candidate bias $b_j^{(t+1)}$ is

$$
\sum_{i=1}^{m}\mathbf 1\!\left[s_{i,j}+b_j^{(t+1)}>\alpha_i^{(t)}\right],
$$

which is monotonically decreasing in the threshold $-b_j^{(t+1)}$. Assuming no ties, setting this count to $q$ makes $b_j^{(t+1)}$ the $(q+1)$-th largest margin $s_{i,j}-\alpha_i^{(t)}$. Since $q/m=k/n$, this is the $(1-k/n)$-quantile of the margins across tokens, giving the QB update

$$
\begin{aligned}
b_j^{(t+1)}&\leftarrow\operatorname{quantile}_{1-k/n}(\mathbf s_{:,j}-\boldsymbol\alpha^{(t)}),\\
\mathbf b^{(t+1)}&\leftarrow\mathbf b^{(t+1)}-\operatorname{mean}(\mathbf b^{(t+1)})\mathbf 1.
\end{aligned}
\tag{14}
$$

The margins subtract the biased cutoff $\alpha_i^{(t)}$ from the raw score $s_{i,j}$, so the old bias enters the update only through the cutoffs, and the second line removes a common offset that leaves Top-$k$ selection unchanged. For causality, the update takes effect only in the next step [Dee24a], i.e., a batch is never routed with a bias derived from itself. Figure 5 illustrates the case $m=8$, $n=4$, and $k=1$, where each expert receives the target load $q=2$. The final bias is frozen at inference. The balanced-assignment derivation is given in Appendix C.

![Quantile Balancing example](../../papers/kimi-k3/figure-05.png)

**Figure 5.** Quantile Balancing with $m=8$ tokens, $n=4$ routed experts, and $k=1$ selected expert per token. The expert-side bias update changes the imbalanced loads $(4,3,1,0)$ into the balanced load $(2,2,2,2)$.

#### Histogram estimation

At scale, the quantile in Eq. 14 spans the full global batch, whose margins number in the millions and are spread across ranks and accumulation steps, so gathering them for an exact quantile is not viable at training time. We instead read each expert's quantile from a histogram of its margins: a single all-reduce sums the per-rank bin counts, and the quantile is recovered from the pooled counts. Because counts are additive, the histogram represents the pooled global batch regardless of how tokens are sharded, so the estimate reflects the whole-batch quantile up to the bin width, at a communication cost of only a few hundred bins per expert. This histogram estimator is the method we use in practice; we give more detailed descriptions of it and its error bound in [§D](#d-histogram-based-quantile-estimation).

### 2.4 Native Vision

Kimi K3 is natively multimodal: text, images, and videos are processed by a single shared backbone within one context, with no post-hoc modality-alignment stage. This design is the architectural foundation of the long-horizon, vision-in-the-loop behavior described in [§1](#_1-introduction). Rendered outputs and the code that produced them live in the same token stream, the model can write code, inspect screenshots or video frames of the result, and iteratively refine visual artifacts-user interfaces, graphics, video-with no cross-model hand-off.

#### MoonViT-V2

A key departure from Kimi K2.5 is that we train Kimi K3's vision encoder, MoonViT-V2, entirely from scratch with next-token prediction. Prior practice, including Kimi K2.5 itself, initializes the vision encoder from a contrastively pre-trained model such as SigLIP, under the premise that pre-trained visual knowledge gives the model a head start. We depart from this practice primarily for training stability. When a pre-trained encoder is attached to the LLM, joint optimization becomes unstable: the SigLIP-initialized MoonViT-3D shows persistently higher gradient norms with frequent spikes, while MoonViT-V2 remains stable throughout training (Figure 6). Training with next-token prediction also allows the encoder's representations to be shaped directly by the language-modeling objective, rather than by a contrastive loss that favors global semantics over fine-grained textual and structural cues. Notably, we find MoonViT-V2 matches the SigLIP-initialized baseline across vision evaluations, indicating that contrastive pre-training is unnecessary as an initialization for multimodal language models at scale.

![Vision-tower gradient norms](../../papers/kimi-k3/figure-06.png)

**Figure 6.** Vision-tower gradient norms in pre-training ablations. Compared with the SigLIP-initialized MoonViT-3D, the from-scratch MoonViT-V2 maintains lower gradient norms with fewer spikes, indicating more stable optimization.

**Architecture.** This training recipe builds on a vision pathway that follows the overall design of Kimi K2.5 [Kim26a, Tea25a]: visual inputs are first encoded by MoonViT-V2 and then mapped by a lightweight MLP projector into the LLM. MoonViT-V2 is a 27-layer vision transformer with roughly 0.4B parameters that adopts RMSNorm and removes all bias terms from its linear and attention projections, a design that further stabilizes the from-scratch optimization above. Images and videos are processed with fully shared parameters, as in MoonViT-3D: attention is factorized into intra-frame spatial and inter-frame temporal passes, and temporal pooling further compresses tokens along the time dimension. Before projection, a pixel-shuffle operation with $2\times2$ downsampling reduces the number of visual tokens by a factor of four, keeping inputs of up to $3584\times3584$ pixels affordable within the 1M-token context.

### 2.5 Per-Head Muon

Following Kimi K2, Kimi K3 adopts Muon [Kel24] as the optimizer for its matrix parameters. For attention projections, we further refine it into a per-head variant: instead of applying Newton-Schulz orthogonalization to the full $\mathbf Q$, $\mathbf K$, and $\mathbf V$ projection matrices, we partition their momentum matrices along the head dimension and orthogonalize each head's block separately. The intuition is that full-matrix orthogonalization treats all heads as a single coupled block, so heads with larger gradient or momentum scales dominate the shared update direction, while smaller-scale heads receive insufficiently normalized updates; per-head orthogonalization equalizes the update scale across heads. In practice, this design yields more balanced learning dynamics across heads and improves training stability at larger scales. It also slightly reduces optimizer overhead, as Newton-Schulz iterations on tall per-head blocks are cheaper than on the full projection matrix.

## 3 Pre-Training

### 3.1 Pre-Training Data

Kimi K3 is pre-trained on a curated corpus spanning four primary text domains-Web Text, Code, Mathematics, and Knowledge-together with a large-scale vision corpus. The vision data covers captions, interleaved image-text documents, OCR, perception, video, and visual coding data. Our data pipelines build on those developed for Kimi K2 [Tea25] and refined in Kimi K2.5 [Kim26a].

**Text data.** Each domain is filtered by a combination of rule-based heuristics, classifier-based quality scoring, and deduplication, with domain-specific sampling rates determined by ablation studies on smaller models. Following the rephrasing recipe of Kimi K2 [Tea25], we rephrase knowledge and mathematics corpora with style and perspective-diverse prompting, chunk-wise autoregressive generation, and fidelity verification against the source documents.

#### Vision data

The vision corpus follows the taxonomy of Kimi K2.5 [Kim26a], combining open-source collections with in-house pipelines for filtering, synthesis, and deduplication. During training, coordinate supervision is provided in both absolute and normalized ($[0,1]$) formats, enabling precise and resolution-robust localization. In addition to classical text-captioned images, we substantially scale up programmatic multimodal data, coupling code snippets with their rendered visuals across domain-specific formats including SVG, 3D assets, Webpage, Game, and CAD schematics.

### 3.2 Scaling Law

Taken together, the architectural, data, and training improvements described in the previous sections define our new model family. Since these changes also alter the optimal training regime, we conduct dedicated scaling-law studies to retune key hyperparameters, including the batch size, learning rate, tokens-per-parameter ratio (TPP), and model shape. Evaluated on held-out OOD validation data, the scaling-law curves in Figure 7 show that these improvements collectively deliver an approximately $2.5\times$ gain in overall scaling efficiency over Kimi K2. Table 1 provides a detailed architectural comparison between Kimi K2 and Kimi K3, highlighting the structural changes that contribute to this improvement.

![Scaling-law curves for Kimi K2 and Kimi K3](../../papers/kimi-k3/figure-07.png)

**Figure 7.** Fitted scaling-law curves for Kimi K2 and Kimi K3. Kimi K3 achieves a $2.5\times$ gain in scaling efficiency over Kimi K2.

![Architectural comparison between Kimi K2 and Kimi K3](../../papers/kimi-k3/table-01.png)

**Table 1.** Architectural comparison between Kimi K2 and Kimi K3.

Our scaling-law study consistently favors cosine decay over Warmup Stable Decay (WSD) [Hu24], leading us to adopt cosine decay as the default learning rate schedule. We compare cosine decay and WSD under a fixed minimum learning rate. Although prior work has reported that WSD can match or even outperform cosine decay, we observe that the two schedules exhibit markedly different optimal hyperparameters. Even under the same model size and training-token budget, their optimal peak learning rates and batch sizes differ substantially. As a result, comparing the two schedules using a shared set of hyperparameters may unfairly favor one simply because those hyperparameters are better aligned with it. To ensure a fair comparison, we conduct an independent scaling-law search for each schedule. Under their respective optimal hyperparameter settings, cosine decay consistently achieves a lower final loss than WSD.

### 3.3 Training Recipe

Kimi K3 adopts a native multimodal training strategy in which language and vision are jointly optimized from the start of training, rather than grafting a vision encoder onto a pre-trained language model through a post-hoc alignment stage. Under this paradigm, visual and textual tokens are interleaved within a single next-token prediction objective, enabling the shared backbone to learn unified multimodal representations from the outset.

We optimize the model using the Per-Head Muon optimizer ([§2.5](#_2-5-per-head-muon)) together with the weight-clipping mechanism introduced in Kimi K2, while adopting QB ([§2.3.3](#_2-3-3-quantile-balancing)) for MoE load balancing. We use a cosine learning rate schedule with a 1% linear warmup. Weight decay is set to 0.1 throughout.

Our pre-training begins with a context length of 8k tokens, which is later extended to 64k tokens in a subsequent training phase.

### 3.4 Long-Context Extension

**Positional encoding.** Kimi K3 uses no explicit positional embedding (NoPE), and instead encodes positional information implicitly through the recurrent gating and decay mechanism of KDA. As a result, the model extrapolates directly to 1M-token contexts without any positional-encoding modification, such as RoPE rescaling or interpolation [Pen23].

**Long-context data.** Long documents and videos from natural sources contain a substantial amount of low-quality content, including near-duplicates, binary blobs, truncated files, video clips, and invalid machine-generated logs. We therefore process them through a dedicated cleaning pipeline that combines exact and fuzzy deduplication, supplemented by perceptual hashing over frames for video, together with heuristic and classifier-based quality filtering, and structural validation. Because genuinely long and coherent documents and videos are scarce relative to short text, we upsample them so that the long-context distribution is not overwhelmed by short sequences during cooldown. Length alone, however, does not confer long-range capability. To address this, we synthesize additional long-context data by carefully permuting and concatenating multimodal documents and sub-tasks, so that the embedded tasks can be solved only by attending to information scattered across the full 1M-token context. This trains the attention mechanism at the intended scale and prevents it from degenerating into local patterns.

**Progressive context extension.** Kimi K3 supports a context window of up to 1 million tokens. We achieve this through extending the context window progressively as training proceeds, following a four-stage curriculum. The window grows from 8K to 64K tokens during pre-training, and from 256K to 1M tokens during the cooldown phase. Concentrating the costly long-sequence computation within a small fraction of the overall training budget keeps the curriculum economical while still allowing the model to adapt gradually to increasingly long-range dependencies. The sequence-dimension partitioning that makes million-token training tractable for the KDA layers is described in [§5.1.2](#_5-1-2-kda-context-parallelism).

## 4 Post-Training

### 4.1 Method

Our post-training pipeline follows a three-stage paradigm: initializing baseline agent capabilities via supervised fine-tuning (SFT), developing specialized domain experts at varying reasoning effort via Reinforcement Learning (RL), and consolidating these domain-specific policies into a single model using Multi-Teacher On-Policy Distillation (MOPD).

#### 4.1.1 Supervised Fine-Tuning

The SFT stage establishes a high-quality cold-start policy for the subsequent RL stage. Building on the SFT pipeline of previous Kimi models [Tea25, Kim26a], we expand the SFT dataset for Kimi K3, substantially broadening its coverage of complex agentic tasks. Specifically, we synthesize data trajectories using domain-specialized models from the prior Kimi series, followed by multi-stage verification and human-in-the-loop annotation. To represent these complex agentic trajectories consistently, we serialize all data with our XTML-based chat template (eXtensible Token Markup Language; see [§F](#f-chat-template) for details). Collectively, these steps yield a large-scale instruction dataset that endows Kimi K3 with adaptive reasoning, precise tool calling, and robust execution in long-horizon agentic scenarios. In addition, we apply quantization-aware training (QAT) from the SFT stage onward, with MXFP4 weights and MXFP8 activations ([§4.1.4](#_4-1-4-deployment-aware-post-training)).

#### 4.1.2 Reinforcement Learning

![RL scaling across public and in-house evaluations](../../papers/kimi-k3/figure-08.png)

**Figure 8.** Scores and average assistant steps across public and in-house evaluations during reinforcement learning. Scaling RL FLOPs consistently increases tool-call steps and improves overall capability.

While SFT provides a solid cold-start foundation, RL is critical to unlocking higher-order reasoning and execution capabilities. Rather than training specialized RL models for individual tasks, we scale RL across three broad domains, each encompassing a wide spectrum of sub-tasks, and train a single expert for each domain at every reasoning effort level: (i) general tasks, spanning general experience, vision, reasoning, faithfulness, search capabilities, and knowledge work tasks; (ii) general agents, spanning long-horizon assistant tasks, deep research, and paragraph-level writing; and (iii) coding agents, spanning software engineering (SWE), coding experience, kernel tasks, and web development. As shown in Figure 8, scaling RL FLOPs consistently improves a variety of capabilities across knowledge, reasoning, vision, general agent, and coding. Crossing these three domain experts with three reasoning-effort levels in $\{\mathrm{low},\mathrm{high},\mathrm{max}\}$ yields nine expert models.

**Algorithm.** To mitigate the long-tail latency that intensifies in long-horizon tasks, we extend the partial rollout scheme from our synchronous RL framework [Sca25, Kim26a]. During the rollout phase of each iteration, we sample $K$ completions for each of $N$ prompts, maintaining an active workload of $N\times K$ trajectories. Rather than waiting for all rollouts to terminate, the generation phase pauses as soon as a fraction $\lambda\in(0,1)$ of trajectories completes, i.e., $\lambda NK$, allowing policy optimization to proceed without execution stragglers. Paused rollouts are enqueued and prioritized for resumption at the start of the next iteration, powered by our sandbox infrastructure ([§5.3.2](#_5-3-2-sandbox-infrastructure)). Once all $K$ responses for a prompt complete, they are immediately dispatched for policy optimization, which follows the algorithm in Kimi K2.5 [Kim26a]. Under our partial rollout scheme, an individual long-horizon trajectory naturally spans multiple iterations, introducing data staleness that threatens training stability. Our policy optimization algorithm inherently tolerates such an extreme off-policy regime through per-token regularization. By constraining policy updates within a localized neighborhood, this regularization enables the algorithm to robustly handle highly stale data and sustains training stability.

**Reasoning Effort RL.** To fine-tune reasoning effort while maximizing token efficiency, we implement a per-problem budget-control mechanism during RL [Kim26a]. We associate each problem $x$ with an initial token budget $b_0(x)$ estimated from the cold-start model, and override the task reward with $-1$ for trajectories whose total token budget $T(y)$ exceeds a scaled threshold $\tau b_0(x)$. For general tasks, $T(y)$ measures the number of thinking tokens, whereas for agentic tasks, $T(y)$ accounts for the cumulative output tokens, including both reasoning traces and tool-call arguments. Training follows a stage-wise curriculum over the budget multiplier $\tau$. We first train a max-budget variant with a relatively large $\tau$, while still capping the maximum budget to suppress excessive overthinking. We then anneal $\tau$ to smaller values to obtain the high- and low-effort expert models. The adjustment of $\tau$ is configured per domain under human-in-the-loop guidance. Trajectories produced by the resulting experts at all reasoning levels are jointly collected for supervised fine-tuning and multi-teacher on-policy distillation.

#### Agentic Generative Reward Model

For non-verifiable general tasks, we adopt an Agentic Generative Reward Model (GRM), retaining the tournament-style group reward with binary comparisons as in Kimi K2.5 [Tea25, Kim26a]. Beyond generic agentic capabilities for enhanced judgment, the agentic judge is required to follow a mandatory protocol: (1) read the outcome, product, or text output; (2) generate a rubric; (3) score each candidate against the rubric; and (4) record the rubric-assigned scores in a scorepad. To mitigate reward hacking toward increasingly verbose outputs, we apply a budget-based verbosity control analogous to the reasoning-effort control above: given an initial verbosity $\ell_0$ estimated from the cold-start model and a multiplier $\sigma$, a candidate whose output length exceeds $\sigma\ell_0$ automatically loses the binary comparison.

#### 4.1.3 Multi-Teacher On-Policy Distillation

We adopt Multi-Teacher On-Policy Distillation (MOPD) to consolidate these domain-specialized capabilities across varying reasoning efforts into a unified model [Lu25, Xia26, Dee26]. During training, for a given domain $d$ and a sampled reasoning effort level $e\in\{\mathrm{low},\mathrm{high},\mathrm{max}\}$, optimization is guided by the corresponding teacher model $\pi_{\mathrm{teacher}}^{(d,e)}$ among the nine experts. Given an input query $x$ and prefix response $y_{<t}$, the per-token on-policy distillation reward evaluated on $y_t$ is

$$
r_{\mathrm{opd}}^d(y_t\mid e,x,y_{<t})=
\operatorname{clip}\!\left(
\operatorname{sg}\!\left[
\log\frac{\pi_{\mathrm{teacher}}^{(d,e)}(y_t\mid x,y_{<t})}
{\pi_\theta(y_t\mid e,x,y_{<t})}
\right],-R_{\max},R_{\max}\right).
\tag{15}
$$

where $\operatorname{sg}(\cdot)$ denotes the stop-gradient operator, and $R_{\max}>0$ is a clipping threshold that constrains extreme advantage signals, thereby stabilizing RL training. This dense reward signal seamlessly integrates into our RL framework, naturally enabling infrastructure-level optimizations such as partial rollout training for long-horizon tasks. While we also experimented with more fine-grained Top-$k$ distillation objectives, we observed no clear advantage in either convergence speed or final performance in our setting.

#### 4.1.4 Deployment-Aware Post-Training

#### MXFP4 Quantization-Aware Post-Training

To reduce memory footprint and serving cost at deployment, we quantize the MoE expert weights -which dominate the model's parameter memory -to MXFP4 [Dar23], with activations computed in MXFP8, while all non-expert components (attention projections, latent MoE projections, shared experts, and MoE routers) remain in higher precision. We perform quantization-aware training (QAT) [Ben18] throughout the entire post-training stage, covering both SFT and RL, so that the model adapts to quantization-induced precision loss. During RL, rollout and training share the same quantization scheme -eliminating the train-inference mismatch.

**Draft Model Fine-Tuning.** Optimizing inference efficiency is crucial for serving complex, long-horizon agentic models. Kimi K3 is pre-trained with a multi-token-prediction (MTP) layer that mirrors the structure of a backbone block. As the draft model of EAGLE-3 [Li25] comprises a single decoder layer whose structure matches the MTP layer, we fine-tune the pre-trained MTP layer into an EAGLE-3-style draft model, with the target model frozen and only the draft layer and its feature-fusion projection updated. Following the training-time test protocol of EAGLE-3, the draft is unrolled for seven steps during training; beyond the first step, where the target-side features of the newest position are unavailable, the draft consumes its own outputs from earlier steps, mirroring the recurrent drafting procedure at inference.

The draft input fuses low-, mid-, and high-level features of the target model, taken from the outputs of the 1st, 4th, and final AttnRes blocks, respectively ([§2.2](#_2-2-attention-residuals)). These features are concatenated and projected to the hidden size by a bias-free matrix $\mathbf W_{\mathrm{E3}}$, initialized as $[\,\mathbf 0\;\mathbf 0\;\mathbf I\,]$ so that the fused representation initially coincides with the high-level feature $\mathbf h^h$, the input on which the MTP layer was pre-trained, and gradually learns to incorporate the low- and mid-level features during fine-tuning.

The speedup of speculative decoding is governed by the per-token acceptance rate $\sum_{x\in\mathcal V}\min(p(x),q(x))$ under lossless speculative sampling, where $p$ and $q$ denote the next-token distributions of the target and draft models. Since minimizing the conventional KL-divergence surrogate does not guarantee maximizing this rate for a capacity-limited draft model, we directly optimize the likelihood-based LK loss [Sam26], the negative logarithm of the acceptance rate itself,

$$
\mathcal L_{\mathrm{LK}}=-\log\sum_{x\in\mathcal V}\min(p(x),q(x)).
\tag{16}
$$

with $p$ and $q$ evaluated at temperature 1 and no auxiliary ground-truth cross-entropy term. Draft fine-tuning follows the post-training QAT configuration ([§4.1.4](#_4-1-4-deployment-aware-post-training)), with MoE expert weights in MXFP4 and their input activations in MXFP8, while non-expert modules remain in higher precision.

### 4.2 RL Task Synthesis and Agentic Environments

The effectiveness of our RL framework relies heavily on rich, diverse, and robustly verifiable environments. To support scalable training across complex long-horizon tasks, we design a series of specialized white-box environments and task synthesis paradigms.

#### 4.2.1 Unified White-Box RL Environment

Training with a single fixed agent harness can cause a model to overfit to a particular tool schema, system prompt, context management mechanism, or interaction protocol. To address this, we develop a unified white-box RL environment that represents an agent harness as a collection of configurable, composable modules, including tool interfaces, system prompts, context management strategies, skills, memories, subagents, and other components. Composing these modules through configuration, the environment can instantiate mainstream harnesses such as Kimi Code [Kim26], Claude Code [Cod26], Codex [Cod26a], OpenClaw [Ope26a], and Hermes [Age26b], as well as entirely new ones. During RL training, we dynamically construct different harness configurations for different task groups, exposing Kimi K3 to diverse combinations of these modules rather than the conventions of any single harness. The same abstraction also readily supports RL across various task domains, providing a scalable foundation for training more general-purpose agents.

#### 4.2.2 Knowledge-Graph-Guided Task Synthesis

**Motivation and overview.** The quality and diversity of post-training tasks are largely determined by their source materials. Retrieval guided by fine-grained concepts surfaces specialized and underrepresented knowledge, while sampling across diverse concepts broadens domain coverage. To control both granularity and coverage at scale, we build a self-evolving, hierarchically organized knowledge graph that agents continuously expand through web-scale exploration across knowledge-intensive and coding domains. Figure 9 illustrates the task synthesis pipeline. The hierarchically organized knowledge graph represents concepts at multiple levels, ranging from broad domains to fine-grained concepts. Related nodes are sampled to form a keyword set that guides the retrieval of publicly available source materials. For each synthesis instance, the system selects a task type and uses the retrieved materials to synthesize a corresponding task.

![Knowledge-graph-guided task synthesis](../../papers/kimi-k3/figure-09.png)

**Figure 9.** Knowledge-graph-guided task synthesis. Related nodes in a hierarchical concept graph guide source retrieval and the synthesis of diverse tasks.

#### Agentic knowledge graph construction

We construct the knowledge graph as a directed acyclic graph through recursive, agent-driven expansion. The expansion process begins with a predefined set of coarse-grained seed nodes. An agent instance is then assigned to each node and performs multiple web searches to investigate the corresponding concept. Before adding new nodes, the agent explores the existing graph to identify equivalent or related concepts, reuse existing nodes where appropriate, and minimize duplication. Edges are always directed from the coarser concept to the finer one, regardless of which endpoint the agent discovers first. Newly added nodes are subsequently assigned to agents for further exploration. A branch stops expanding when the assigned agent determines that the current concept is sufficiently atomic.

#### Material retrieval and task synthesis

To target a desired distribution across domains and task types, the system samples nodes at varying levels of granularity, either individually or in related combinations. Keywords derived from the sampled nodes are combined with contextual information from their ancestors in the knowledge graph to formulate web queries. The retrieved real-world materials are assembled so that a synthesis agent produces training tasks of various task types.

#### 4.2.3 Verifiable Problems in Agentic Environments

We train Kimi K3 on verifiable problems in agentic environments; representative examples include multi-step complex information searching, where the model plans its research, gathers evidence from the web step by step, and produces a verifiable answer; the real day-to-day work of professionals, such as investment banking, data analysis, and legal practice, where the model decomposes a complex request, operates domain tools in a sandbox, and completes a deliverable over dozens to hundreds of steps; and multi-step verifiable visual reasoning over STEM problems, visual puzzles, and chart understanding. Each visual-reasoning trajectory is generated in an agent environment equipped with a Python interpreter in an isolated sandbox: the model iteratively writes and executes code to crop, zoom, or transform the input image, perform precise computation, or verify intermediate results, and receives the execution outputs, including generated images -as new observations over multiple interaction steps. As the model learns to perform more image operations and collect more observations, its performance on complex visual reasoning tasks steadily improves.

#### 4.2.4 Kernel Optimization Tasks

To strengthen Kimi K3's GPU kernel optimization capabilities, we build a large-scale suite of kernel tasks ranging from single-operator kernels to fused mega-kernels, sourced from high-quality GitHub repositories such as Flash Linear Attention [Yan24]. The suite spans diverse GPU programming approaches, such as CUDA, Triton, CuTe DSL, Gluon, ThunderKittens [Ben25], and TileLang [Wan25], and covers widely used GPU architectures and numerical formats including BF16, FP8, and FP4. Rewards evaluate both correctness and performance: each kernel provides a PyTorch reference implementation, and solutions exceeding a predefined numerical error threshold receive zero reward. Performance is scored against an expert implementation, where matching it yields a reward of 0.5 and approaching the hardware roofline increases the reward toward 1. To ensure that rewards reflect genuine optimization, we develop a hacking-detection system that penalizes reward-hacking strategies such as CUDA graph replay, input caching, and precision reduction, and we continuously extend it with new safeguards as new hacking strategies are observed during Kimi K3's development.

#### 4.2.5 Personal Assistant Tasks

For long-horizon personal assistant tasks, we develop realistic mock implementations of widely used applications, such as Gmail, Notion, Slack, and Canvas. They preserve the core semantics of their real-world counterparts while enabling reproducible, large-scale interaction without external APIs or rate limits. Building on these mock applications, we design complex tasks inspired by real-world professional workflows in scenarios like human resources, legal services, and finance. In each task, the agent operates in a persistent, evolving environment over multiple simulated days and encounters dozens of interdependent events distributed across applications. A single rollout may involve up to thousands of tool calls and millions of context tokens. Each event carries its own evaluation criterion, assessed by deterministic rules or LLM-based evaluators. The initial workspace is constructed by agents that autonomously search the web for reference materials and transform them into a coherent, task-relevant environment. We also extend our RL framework to support such living environments, modeling complex event streams and the induced world-state transitions.

#### 4.2.6 Autonomous Execution Tasks

We introduce Autonomous Execution Tasks (AET), an environment paradigm that trains long-horizon agent intelligence through verify-in-the-loop optimization. Each task specifies an initial state, a constrained goal, a tool-based action space, execution budgets, and an independent verifier. Agents see only the objective, context, constraints, and verification interfaces, without reference trajectories or predefined procedures, and must autonomously perform task decomposition, tool selection, planning, error recovery, and termination. Rewards are grounded in the verifier's evaluation of the final environment state rather than the agent's self-reported completion. We design multiple types of verifiers that support diverse environments, including black-box system replication (Figure 10), quantitative factor discovery, and tax auditing.

![Autonomous execution task completion curves](../../papers/kimi-k3/figure-10.png)

**Figure 10.** Completion curves on Camera Repair Management System, a black-box system replication task in which the agent reconstructs a hidden 3D-camera repair system as a web application through oracle queries.

In each environment, agents iteratively submit solutions, receive verifier feedback, and refine their strategies, training a general loop of hypothesizing, acting, analyzing feedback, and adapting. Reward hacking is mitigated by isolating agents from verifiers, pairing public verifiers that offer diagnostic feedback with hidden verifiers that evaluate held-out scenarios, and applying penalty-based rewards under limited submission budgets.

#### 4.2.7 Web Development Tasks

We construct a diverse suite of expert-curated web development tasks covering typical scenarios. Inputs range from one-line scene descriptions to multi-paragraph specifications; artifacts span websites, interactive games, 3D/WebGL scenes, data visualization, SVGs, and full-stack applications. Every task runs in a containerized sandbox and is rolled out under diverse agent scaffolds rather than a single fixed harness, to promote cross-scaffold generalization. Rewards consist of two components: deterministic checks and model judging by an internal reward model. Deterministic checks functionally test application behavior, and score structural and pixel-level similarity for tasks that replicate a reference. The reward is zeroed when a project fails to build, runs with errors, or fakes rather than implements the artifact. Model judging uses other models to perform source code inspection or to look at and interact with the output artifact.

## 5 Infrastructure

Kimi K3 combines three system challenges rarely encountered in a single model: hybrid KDA attention, 3T-class sparse multimodal training and inference, and million-token agentic workloads. Our infrastructure is co-designed with these challenges across the model lifecycle. At the architecture level, high-performance KDA kernels and Context Parallelism make the recurrent formulation efficient within and across devices, in both training and inference. During pretraining, balanced expert execution, reduced memory footprint, and communication-overlapped scheduling sustain high utilization at scale. During 1M-token agentic RL, hierarchical state management and resumable sandbox execution preserve long trajectories across iterations. Finally, state-aware KDA prefix caching, specialized inference kernels, and cache-and budget-aware scheduling translate these efficiencies into predictable production serving.

### 5.1 Algorithm-System Co-Design for KDA

KDA replaces the growing key-value cache of softmax attention with a fixed-size recurrent state $\mathbf S\in\mathbb R^{d_k\times d_v}$ ([§2.1.1](#_2-1-1-kimi-delta-attention)), whose serial update poses challenges in parallel execution, in exchange for a fixed-size state that is cheap to transfer and reuse. The designs below address the first property and exploit the second at two levels of execution, with fused kernels within a device and KDA Context Parallelism across devices.

#### 5.1.1 KDA Kernels across Regimes

The serial dependence of the KDA state is at odds with the GPU's preference for wide, uniform parallelism, and it manifests as a different bottleneck in each execution regime. We design a dedicated kernel for each regime.

**Chunkwise kernel for training and prefill.** The chunkwise form of KDA is parallel within each chunk but serial across chunks, since the recurrent state must propagate from chunk to chunk. Executed naively, these two phases alternate, leaving the SMs idle during the serial propagation. We therefore develop FlashKDA [Che26a], a CUTLASS-based chunkwise kernel that overlaps intra-chunk computation with cross-chunk state propagation. The kernel decomposes the work into token-parallel stages and a head-parallel recurrence, each scheduled and tuned independently, and substantially outperforms the Triton reference implementation. FlashKDA serves both training and inference prefill and is auto-dispatched as a backend of flash-linear-attention [Yan24].

**Intra-device context parallelism for long-context prefill.** Tensor parallelism partitions heads across devices but never shortens the recurrence, so under pure TP deployment, prefilling an ultra-long sequence leaves most SMs idle when each rank holds only a few heads. The key observation is that the state transition of each segment can be evaluated independently of the incoming state and composed exactly afterward. An automatic SM-level context-parallel (CP) planner [Wan25a, Yan24] therefore partitions the sequence across the SMs of a single rank, evaluates the segment transitions in parallel, and merges them to recover each segment's exact initial state. In contrast to the cross-device KCP of [§5.1.2](#_5-1-2-kda-context-parallelism), this parallelism is entirely intra-device and incurs no cross-device communication.

KDA decoding presents challenges distinct from those encountered during training and prefill. We discuss these challenges in detail in [§5.4.2](#_5-4-2-high-performance-kernels).

#### 5.1.2 KDA Context Parallelism

The communication overhead of context parallelism differs fundamentally between softmax and linear attention. Softmax attention requires ranks to exchange key-value blocks whose size grows with the sequence length [Liu23]. Linear attention instead carries the preceding context in a fixed-size recurrent state $\mathbf S\in\mathbb R^{d_k\times d_v}$. Prior context-parallel methods exploit the additive recurrence of vanilla linear attention by computing, on each rank, the state that the local tokens generate from $\mathbf S=0$ and summing these local states over the preceding ranks to recover the incoming state [Sun24, Sun25].

This direct summation, however, is insufficient for KDA. Recall from Equation 1 that KDA updates its state as $\mathbf S_t=\mathbf M_t\mathbf S_{t-1}+\beta_t\mathbf k_t\mathbf v_t^\top$, where $\mathbf M_t:=(\mathbf I-\beta_t\mathbf k_t\mathbf k_t^\top)\operatorname{Diag}(\boldsymbol\alpha_t)$. KDA's delta rule applies the token-dependent matrix $\mathbf M_t$ to the incoming state before adding the current write. Consequently, the effect of a local sequence segment depends on the state entering that segment and cannot be determined from the state computed with $\mathbf S=0$ alone.

To preserve this dependence, we introduce KDA Context Parallelism (KCP), which decomposes the effect of each segment into two locally computable quantities: a cumulative transition acting on the incoming state and a state generated locally from zero. Following the chunkwise notation of [§2.1.1](#_2-1-1-kimi-delta-attention), we write $\mathbf S_{[i]}^t$ for the recurrent state within the segment of rank $i$ after $t$ local tokens, so that $\mathbf S_{[i]}^{T_i}$ denotes the state leaving rank $i$ and entering rank $i+1$. We write $\widetilde{\mathbf S}_{[i]}^t$ for the state of the same recurrence started instead from $\mathbf S=0$. For an arbitrary state entering the $(i+1)$-th of $P$ context-parallel ranks, the state after $t$ local tokens is

$$
\begin{aligned}
\mathbf M_{[i+1]}^{t\leftarrow1}&:=\prod_{r=1}^{t}\mathbf M_r\in\mathbb R^{d_k\times d_k},\\
\mathbf S_{[i+1]}^t
&=\widetilde{\mathbf S}_{[i+1]}^t+\mathbf M_{[i+1]}^{t\leftarrow1}\mathbf S_{[i]}^{T_i}\\
&=\widetilde{\mathbf S}_{[i+1]}^t+\mathbf M_{[i+1]}^{t\leftarrow1}
\sum_{j=1}^{i}\left(\prod_{l\leftarrow j+1}^{i}\mathbf M_{[l]}^{T_l\leftarrow1}\right)
\widetilde{\mathbf S}_{[j]}^{T_j}
\in\mathbb R^{d_k\times d_v}.
\end{aligned}
\tag{17}
$$

where $\mathbf M_{[i+1]}^{t\leftarrow1}$ denotes the cumulative transition of the first $t$ local tokens. The first term contains the state generated by the local tokens, whereas the second term propagates the context from preceding ranks through the local KDA updates. At $t=T_{i+1}$, both $\mathbf M_{[i+1]}^{T_{i+1}\leftarrow1}$ and $\widetilde{\mathbf S}_{[i+1]}^{T_{i+1}}$ can be computed using only the local tokens, before $\mathbf S_{[i]}^{T_i}$ is available, and are the fragments each rank exchanges with the others.

The summation in Equation 17 shows that every state is composed purely from locally computed fragments. These rank-level updates compose associatively, so the incoming state of each rank can be recovered by a prefix scan [Mar18]. Each rank first computes $\mathbf M_{[i]}^{T_i\leftarrow1}$ and $\widetilde{\mathbf S}_{[i]}^{T_i}$ locally, then exchanges both tensors with one all-gather [Yan24]. After the all-gather, rank $i+1$ reconstructs $\mathbf S_{[i]}^{T_i}$ by processing preceding fragments of the same document in order, starting from $\mathbf S=0$ and applying $\mathbf S\leftarrow\mathbf M_{[j]}^{T_j\leftarrow1}\mathbf S+\widetilde{\mathbf S}_{[j]}^{T_j}$ at each fragment. Therefore, KCP requires only a fixed-size all-gather for recurrent-state synchronization and achieves linear compute scaling.

### 5.2 Infra for 3T-class Pre-Training

Kimi K3 pre-training combines Pipeline Parallelism (PP) with virtual stages (VP) [Hua19, Nar21], Expert Parallelism (EP) [Lep20], ZeRO-1 Data Parallelism [Raj20], Pipeline ZeRO-2 gradient sharding [Zen26], and Context Parallelism (CP, [§5.1.2](#_5-1-2-kda-context-parallelism)) [Sam23].

The MoE layers employ shared experts replicated across EP ranks, and the all-to-all communication for expert dispatch and combine is overlapped with computation to hide its latency.

Natively multimodal pre-training at the 3T-class poses three critical problems: (i) token loads are imbalanced across EP ranks; (ii) activations, gradients, and optimizer states exceed the memory budget; and (iii) the vision encoder's highly variable computation is exposed on the critical path. The following subsections address these problems in turn: perfectly balanced expert-parallel MoE training ([§5.2.1](#_5-2-1-perfectly-balanced-expert-parallel-moe-training)), memory-efficient training ([§5.2.2](#_5-2-2-memory-efficient-training)), and multimodal encoder optimization ([§5.2.3](#_5-2-3-multimodal-encoder-optimization)). Fig. 11 illustrates the resulting execution schedule.

![Overlapped pre-training execution schedule](../../papers/kimi-k3/figure-11.png)

**Figure 11.** Computation, communication, and offloading overlapped in different pipeline-parallel phases.

#### 5.2.1 Perfectly Balanced Expert-Parallel MoE Training

In conventional EP schemes, token loads are imbalanced across ranks. The resulting computational imbalance degrades training throughput, and the dynamically varying shapes of routed-expert activations cause substantial memory fragmentation. We therefore propose MoonEP ([repository](https://github.com/MoonshotAI/MoonEP)), an EP scheme that achieves perfect load balance with dynamic redundant experts. MoonEP preserves the overall computation flow of conventional schemes such as DeepEP [ZhaWeb] and additionally introduces online planning and migration of redundant experts. In the forward pass, we plan the redundant experts from the router outputs of the current micro-batch and layer and prefetch them before the routedexpert computation. In the backward pass, we stage their gradients in a local reduce buffer and, once the computation completes, reduce them back to the gradient buffers of their home ranks.

**Perfect balance with bounded redundant experts.** MoonEP requires every rank to receive exactly $S\times K$ tokens, where $S$ is the sequence length and $K$ is the number of experts selected per token, so that all ranks perform identical amounts of computation. The key question is how many redundant experts suffice to guarantee such a balance. Let $E$ be the number of experts and $R$ the EP size. We prove that a balanced plan always exists with at most $E/R$ redundant experts per rank and that this bound is essentially tight (Appendix E). Reserving $E/R$ redundant-expert slots per rank therefore guarantees that planning always admits a feasible solution, so training is never interrupted. In contrast, prior work such as ECHO [Yan26] and UltraEP [Wei26] presets the number of redundant experts or imposes a per-rank token cap. Training is then forced to stop whenever no feasible plan exists within the cap, and the cap itself requires manual tuning while still leaving residual imbalance.

**Online planning.** Computing the exact optimum at every training step is prohibitively expensive. We therefore compute exact solutions offline with integer linear programming (ILP) for representative cases as references and design a GPU planning kernel that is near-optimal, incurs negligible overhead, and always respects the $E/R$ upper bound.

**Zero-copy communication.** Perfect balance also simplifies the communication path. We implement a fused permute/unpermute operator in which the planning kernel precomputes the destination of every token, so tokens are sent directly to their expert-grouped positions on remote ranks, and views of the communication buffer are returned directly to the computation, eliminating intermediate copies. Under worst-case imbalance, supporting the same copy-free data path in DeepEP requires a communication buffer of size $S\times K\times R$, whereas MoonEP requires only a fixed $S\times K$ buffer owing to the perfect balance.

**Static computation shapes.** Conventional implementations synchronize dynamic shapes before launching expert computation, stalling the pipeline between layers. With perfect balance, every rank receives exactly $S\times K$ tokens and the computation shapes of all layers are statically known. This eliminates the per-layer MoE host synchronization and alleviates the host-side kernel-launch overhead.

**Expert-GEMM scheduling and overlap.** Even with the aggregate load perfectly balanced across ranks, the per-expert token counts within each rank remain skewed, and a fixed-order, workload-oblivious schedule turns this skew into an imbalanced makespan across SM workers. We therefore schedule the routed-expert GEMM with a workload-aware scheduler that adapts its parameters to the current token distribution before launch and keeps them fixed during execution.

A lightweight heuristic selects these parameters using an analytical cost model of hardware metrics, with key coefficients calibrated through offline autotuning. For the shared experts, we dispatch their GEMMs to a separate stream so that they overlap with other kernels.

#### 5.2.2 Memory-Efficient Training

**Unified activation manager.** We design a unified storage abstraction for activations, in which every tensor saved for the backward pass is associated with a pluggable storage backend. Recomputation, quantization, and offload/remoteoffload are merely storage policies under this abstraction and can be freely composed at tensor granularity; policies are declared via lightweight annotations on tensors, fully decoupled from the model code. Recomputation is performed at function granularity, which supports cross-layer recomputation. In our implementation, all GPU memory is allocated on the main compute stream and managed within a single memory pool, avoiding multi-stream fragmentation and host-bound overhead; activations are prefetched back at layer granularity and overlapped with computation, introducing negligible extra overhead. In Kimi K3, most activations use block-wise FP8 quantization [Tea25, Dee24a] combined with offload/remote-offload, and element-wise operators are configured with recomputation.

#### Memory-efficient MoE

In the native MoE implementation, the gradient computation of `permuted_probs` depends on the forward output `output`. Inspired by SonicMoE [Guo25a], we rewrite this gradient through a mathematical transformation into a form that depends only on the intermediate activation `act_output` and the upstream gradient `doutput`, eliminating the backward dependency on `output` at the cost of an additional lightweight element-wise computation. Furthermore, in the forward pass of the group GEMM, we save only the input of the dispatch operation; during the backward pass, the input of the group GEMM is recovered by recomputing dispatch. As shown in Fig. 11, the communication introduced by this recomputation can be overlapped with part of the group-GEMM backward computation, eliminating this portion of activation storage at a negligible cost.

#### Memory-efficient Attention residual

For the attention residual, we design a companion optimization based on Block AttnRes. The block representation is generated once at the boundary layer and shared by all subsequent layers, residing directly on the GPU. The AttnRes computation is entirely wrapped with checkpointing, so the activation saved for the backward pass at each layer is identical to that of the standard residual architecture. For pipeline parallelism, we adopt cache-based pipeline communication [Tea26], in which only newly generated blocks are incrementally transferred between stages and released as soon as the micro-batch finishes, reaching the theoretical lower bound on memory footprint.

**Balancing activations across PP ranks.** Under interleaved 1F1B pipeline parallelism, activations are unevenly distributed across PP ranks due to pipeline warmup, and the number of resident activations decreases as the PP rank increases. To avoid out-of-memory (OOM) errors, we remotely offload activations to the memory of other PP ranks using the Mooncake Transfer Engine [Qin24], achieving balanced activation memory across PP ranks.

**Pipeline ZeRO-2 gradient sharding and offloading.** Beyond activations, we use Pipeline ZeRO-2 gradient sharding [Zen26] to shard gradients across data-parallel (DP) ranks. Furthermore, we store the sharded gradients in CPU memory to reduce peak GPU memory usage, while keeping the double grad buffer on the GPU. After gradients are reduced across DP ranks into the double grad buffer, they are accumulated into the CPU shards.

#### P2P-based Muon orthogonalization

The distributed optimizer shards parameters evenly across DP ranks, whereas the Newton-Schulz orthogonalization in Muon requires the full parameter matrix, necessitating a communication step to gather complete parameters before each update. The naive approach performs an all-gather over the entire parameter buffer on every rank [Liu25], which incurs a substantial memory footprint on top of making communication the primary bottleneck at scale. Instead, each rank retrieves only the shards of its locally owned parameters via peer-to-peer (P2P) communication with the corresponding owner ranks, eliminating the full-parameter buffer and reducing both memory usage and communication volume. Communication and computation are further pipelined at the granularity of model-chunk buffers, hiding the communication overhead.

#### 5.2.3 Multimodal Encoder Optimization

**Dynamic CP in multimodal encoder.** In long-context multimodal training, large images and long videos substantially increase the computation time of the vision encoder and cause significant load imbalance across devices. To address this, we extend context parallelism to such large samples. A single large image is partitioned along the patch dimension across multiple devices, and attention is computed by gathering key-value pairs (gather-KV) across CP ranks. In addition, we divide each CP group into several sub-CP groups and distribute multiple large images across them in a load-balanced manner, preventing the communication fraction from growing with scale. This reduces both the encoder latency of large visual samples and the cross-device load imbalance, allowing the remaining encoder computation to be hidden in pipeline bubbles.

#### Encoder computation in PP bubbles

In Kimi K2.5, we introduced the Decoupled Encoder Process (DEP) [Kim26a], which splits ViT and text training into separate stages and balances vision forward and backward passes across PP stages. We observe that, under the interleaved 1F1B pipeline schedule, the text forward passes of the first PP micro-batches are all scheduled at the very beginning, while the text backward passes of the last PP micro-batches finish only at the very end. We therefore further decompose the ViT computation [Val26a]. The ViT forward passes of the first PP micro-batches are executed synchronously upfront, the remaining forward passes are scheduled into pipeline bubbles, and the backward passes are handled analogously. As a result, most of the ViT computation is hidden within pipeline bubbles, largely eliminating the effective overhead of the vision encoder.

### 5.3 Infra for 1M Agentic RL

Scaling agentic RL for a model as large as Kimi K3 to million-token contexts under a bounded compute budget makes resource efficiency a first-order goal. We therefore develop long-context RL infrastructure for efficient training and rollout, together with high-performance, resumable sandboxes for long-horizon environment interaction.

#### 5.3.1 Long-context RL infrastructure

We adopt co-located RL training [Tea25] to keep each 1M-context Kimi K3 RL experiment within a few hundred GPUs, and use partial rollouts [Sca25] to reduce tail latency from ultra-long trajectories. This design improves hardware utilization, but long-context rollouts introduce extra DRAM demand for KV-cache retention, which competes with training-side states. Further, achieving high efficiency for both prefill and decoding requires careful prefix management and request scheduling.

**External KV cache pool.** At 1M-context multi-step rollout, a prefix KV-cache miss is extremely expensive. Partial rollout exacerbates this issue at the beginning of each iteration, due to many unfinished long prefill requests from the previous iteration arriving at the same time. Speculative decoding further accelerates request turnover within relatively fixed tool-call intervals, increasing prefix-block churn. These issues can trigger preemption and lower the cache hit rate, which is critical for long-context RL.

We therefore decouple prefix retention from GPU residency with a write-back design. Active decoding blocks remain in GPU KV cache, while reusable idle prefixes are written back to an external KV cache pool in CPU DRAM only when it is evicted from GPU, and is prefetched back before the next reuse. KDA states are offloaded and prefetched together with the corresponding MLA KV cache blocks, keeping their lifecycles aligned. Compared with a write-through strategy, this policy incurs CPU DRAM usage and transfer bandwidth only for prefixes that leave the active decode path, avoiding redundant CPU copies of blocks that are still resident and active on GPU.

To provide sufficient DRAM for the external pool, we offload training states (model weights and optimizer states) to NVMe after a training iteration finishes. After a rollout iteration, the pool is released to avoid contention with training workloads.

#### Rollout auto-throttling scheduler

In multi-step rollout, contexts grow progressively as the trajectory advances, making fixed concurrency based on the full-trajectory average length both hard to estimate and overly conservative early on. Conversely, setting concurrency too high creates KV cache pressure in later stages and can trigger preemption. We therefore design an auto-throttling mechanism at the LLM request scheduling layer, using runtime signals such as active request count, queued request count, and KV cache utilization to dynamically control how many requests are sent to the inference engine. This keeps early rollout well utilized while reducing concurrency as KV cache pressure rises, avoiding both under-saturation and overload without manual tuning.

**Gradient-buffer reuse for non-policy model forwarding.** RL loss computation often requires forward-only nonpolicy models, such as reference models, whose weights are too large to keep resident on GPU. We keep these weights in CPU memory and materialize them only when needed, backing their parameter tensors with the policy model's FP32 gradient-buffer storage. This reuses existing GPU memory without extra allocation or fragmentation, and remains safe because the buffers are overwritten when real gradients are later computed.

With ZeRO-2 gradient sharding and offloading ([§5.2.2](#_5-2-2-memory-efficient-training)), each GPU retains gradient buffers for only two VPP chunks in Kimi K3 RL training. We stream reference weights into these slots chunk by chunk: one slot is used for the current forward computation while the other prefetches the next chunk, hiding copy overhead without increasing GPU memory.

#### 5.3.2 Sandbox Infrastructure

We employ multiple sandbox runtimes to support the diverse requirements of Kimi K3 post-training and evaluation, including a traditional container-based runtime, a GPU sandbox runtime, and, most notably, a new microVM-based sandbox runtime called AgentENV.

AgentENV ([repository](https://github.com/kvcache-ai/AgentENV)), developed in collaboration with our partners, is a sandbox system specifically designed for agentic AI workloads. It is built around three core design goals:

- **High-fidelity isolated sandbox runtime.** As agents become more capable and tasks more difficult, they tend to explore more aggressively and may even attempt reward hacking. On the one hand, this poses unique security challenges: in our early experiments with traditional container-based sandbox runtimes, we observed several kernel panics and deadlocks caused by unintended agent operations. On the other hand, we want to permit as much exploration as possible so as not to constrain agent capability, and complex tasks require a sandbox close to a real-world environment -for example, agents should be able to mount disks, run containers, or even launch virtual machines at will. By running isolated microVMs with Firecracker [Aga20], AgentENV provides a level of isolation and fidelity that container-based runtimes cannot match.

- **Flexible sandbox life-cycles for agentic RL.** At the low level, AgentENV supports incremental checkpointing and resuming of sandbox states, where only memory pages dirtied since the last checkpoint are saved during checkpointing, achieving checkpoint and resume latencies as low as 133 ms and 49 ms, respectively. On top of this, AgentENV provides three high-level operations that help improve agentic RL efficiency. (a) Pause and Resume: a paused sandbox consumes no memory or CPU resources; a sandbox can therefore be paused while the agent is waiting for the model's inference result, which can account for as much as 98% of the sandbox lifetime. (b) Fork: fork creates a new sandbox from the exact state of the original one while keeping the original running, which is useful for reward judging without side effects. (c) Snapshot: snapshots of a sandbox can be saved at regular intervals for error recovery.

- **High efficiency and high density.** In our workloads, tens of thousands of sandboxes, each with a unique set of images, may need to be created within seconds. We adopt OverlayBD [Li20] as the image format, together with a custom ublk driver implementation, storage-layer sharing, and P2P transport, achieving sub-second launch latency at large scale. We further reduce memory usage with copy-on-write memory and page-cache optimizations, achieving a memory overcommit ratio of up to $6.5\times$ in real workloads.

Throughout Kimi K3's training and evaluation, a total of 51,219,741 sandboxes across 1,505,678 images were created.

### 5.4 Inference and Online Serving

Serving Kimi K3 exposes the same challenges from the production side: the hybrid KDA-MLA architecture maintains two fundamentally different caches that must be managed jointly at million-token contexts, its new modules and highly sparse experts demand kernels tailored to each, and production traffic mixes requests whose per-request cost spans three orders of magnitude. The designs below address these challenges at three levels. At the engine level, a KDA-aware prefix cache packs the fixed-size recurrent state into the same paged pool as the MLA KV cache and keeps long prefixes reusable across requests. At the device level, dedicated kernels for KDA decoding, Block AttnRes, and the sparse latent MoE minimize per-token latency and memory traffic. At the fleet level, cache-aware affinity scheduling and budget-based admission control translate these efficiencies into predictable serving.

#### 5.4.1 KDA-Aware Prefix Cache Management

The hybrid architecture in Kimi K3 complicates prefix caching: the KDA recurrent state and the MLA KV cache differ fundamentally in size and lifetime, yet a cached prefix is reusable only when both can be restored together at the same boundary. We therefore design a KDA-aware prefix cache that manages the two cache types jointly-from a unified paged layout to fine-grained prefix reuse and consistency under concurrent scheduling-keeping million-token prefixes cheap to retain and reusable across requests.

**Unified cache layout for hybrid KDA-MLA attention.** Each Kimi K3 block consists of three KDA layers and one Gated MLA layer, whose caches differ fundamentally. The MLA KV cache grows with sequence length and is paged per token, whereas the KDA recurrent state is fixed in size with a single copy per request. Maintaining a separate manager for each would duplicate the allocation, eviction, and transfer logic. We therefore pack KDA states into the same paged block pool as MLA KV, unifying pages to the same byte size so that both page types share one implementation of allocation, reference counting, and eviction. Within a page, the states of all heads are stored contiguously head by head, so that each head's byte stream is self-contained and serves as the minimal unit of cross-node transfer. Under prefill/decode disaggregation, when prefill and decode nodes adopt different TP degrees, re-layout is performed on the transfer path with zero GPU-side reshuffling. This asymmetry proved useful during development: any type-confused access yields garbage rather than plausible data -a zero-overhead sanity check on the pooled layout.

#### KDA prefix cache optimization

Block-hash-based prefix caching reuses the KV cache at the granularity of one physical block: only complete blocks are hashed, so only block-aligned prefixes are reusable. This coupling breaks down in Kimi K3. Block-hash matching requires one block size shared by all layers, and a prefix hit is reusable only if the KDA state at the hit boundary has been persisted. A KDA layer maintains a single large recurrent state per sequence rather than per-token entries, so state snapshots are affordable only at sparse boundaries; the shared block size is therefore forced to 1024-6144 tokens and, since hashing is tied to the storage block, the hash granularity as well, although MLA's per-token entries alone would tolerate much finer blocks. At such a coarse granularity caching is nearly useless: requests shorter than one block can never be reused, and chunked prefill exports no cacheable prefix until it crosses a full block boundary.

We therefore decouple the two granularities. Prefix hashing runs on fine hash blocks (e.g., 512 tokens) inside MLA pages, while the physical block remains the coarse allocation unit. Alignment runs the other way for KDA: checkpoints of the recurrent state are saved only at a sparse subset of MLA's hash endpoints, the only positions a lookup can ever reference.

![Fine-grained KDA-aware prefix caching](../../papers/kimi-k3/figure-12.png)

**Figure 12.** Fine-grained prefix caching within a 6144-token physical cache block. A request reuses five 512-token MLA hash blocks and the KDA checkpoint at boundary $B=2560$, then resumes prefill without recomputing $[0,B)$.

During prefill, a partially filled MLA page is registered in the prefix-cache index under the chained hash of its last complete hash block, where each hash covers all preceding hash blocks so that matching an endpoint certifies the whole prefix up to it; the registered endpoint advances as the page fills. Meanwhile, after each forward pass, the KDA kernel persists the recurrent state at the last hash-aligned position processed. Checkpoints are large, so intermediate checkpoints superseded as the request advances are recycled, while those at conversation-turn boundaries are retained for cross-request reuse. Cached checkpoints are read-only snapshots: a hit restores the state by copying it into the request's private running state before the next forward pass, and new checkpoints are written to fresh slots, so a checkpoint visible to other requests is never mutated in place.

Lookup proceeds in two stages (Figure 12). The MLA stage matches whole physical blocks by chained hash and, at the first missing block, falls back to the hash endpoints inside it, so partially filled pages remain hittable. The KDA stage then requires a checkpoint at the candidate boundary in every KDA cache group, each of which maintains an independent recurrent state. The hit is the longest boundary satisfying both stages, always a multiple of the hash block and never required to be a multiple of the physical block. In Figure 12, a request whose first 2800 tokens match the cached prefix hits at $B=2560=5\times512$, deep inside a 6144-token physical block, and resumes prefill from token $B$ instead of recomputing $[0,B)$.

#### Consistency under concurrent scheduling

The remaining design points are each dictated by a concrete failure mode of sharing partially filled blocks, in a setting where a hit block is at once a shared cache entry and the growth point of a private request, and where the MLA and KDA cache groups must agree on every hit boundary. First, all cache groups draw blocks from one shared free list, so allocating a private copy for one group could evict a block that another group has just hit; every hit block is therefore pinned across all groups before anything is allocated. Second, the copy into the private block executes on the GPU immediately before the forward pass, so a block allocated or registered within the current scheduling step would still hand the previous owner's bytes to a reader; such blocks are excluded from matching until their copies land. Third, a checkpoint can restore a request only if it exists in every KDA group, so evicting one group's checkpoint atomically invalidates its siblings -a checkpoint is either hittable in every group or in none. With these mechanisms, every registered state always corresponds to exactly its declared token prefix, and prefix caching for hybrid KDA-MLA models reaches the same generality as for full-attention models: any shared prefix is reusable at any 512-token boundary, independently of request length, chunking, or scheduling interleaving.

#### 5.4.2 High-Performance Kernels

Kimi K3 introduces several new architectural modules: KDA ([§2.1.1](#_2-1-1-kimi-delta-attention)), Block AttnRes ([§2.2](#_2-2-attention-residuals)), and Stable LatentMoE ([§2.3](#_2-3-stable-latentmoe)). We optimize the kernel implementation for each.

**KDA.** Compared with KDA prefill ([§5.1](#_5-1-algorithm-system-co-design-for-kda)), KDA decoding presents a distinct set of challenges: the primary bottleneck shifts from exploiting parallelism to efficiently managing the evolving recurrent state, which is updated in place at every decoding step. This in-place update becomes problematic in MTP-based speculative decoding: if verification rejects a subset of the drafted tokens, the state has already advanced beyond the last accepted token and cannot be trivially rolled back. Maintaining a state snapshot for each draft position would enable rollback, but would also multiply state traffica cost that dominates at the large batch sizes typical of online serving.

The state after any accepted draft prefix, however, is fully determined by the projected inputs of the draft tokens, which are far smaller than the state itself. We therefore cache only these projected inputs, rebuild the states of accepted tokens on-chip, and write back the states of the verified and bonus tokens, a design independently proposed in the concurrent work ReplaySSM [Dao26]. The replayed tokens, the bonus token, and the next draft window share one recurrent loop inside a single fused kernel covering short convolution, input normalization, gating, the KDA recurrence, and output normalization. Verification latency grows sub-linearly with the number of tokens verified and remains below that of state-caching baselines. Because the projection caches never leave the decode stage, prefix caching and prefill-decode disaggregation operate on the same payload as in non-speculative serving.

**Block AttnRes.** Block AttnRes [Tea26] follows a two-phase schedule: a batched inter-block pass reads the cached block representations once per block, after which each layer folds in the intra-block partial sum through an online-softmax merge [Mil18]. Memory access accounts for a substantial fraction of the cost of these kernels in both prefill and decoding, so our optimizations in both stages focus primarily on memory efficiency.

For prefill, materializing the block representations on every tensor-parallel (TP) rank would incur substantial redundant memory consumption. We therefore adopt sequence parallelism (SP) for activations: the TP all-reduce is decomposed into a reduce-scatter and an all-gather, with the intra-block kernel inserted between the two collectives, operating on the sequence-sharded hidden states so that the block representations of each token are materialized on exactly one rank. This eliminates the additional memory consumption and reduces the I/O overheads of Block AttnRes during prefill.

For decoding, we launch the inter-block kernel on a side stream so that it overlaps with independent computation on the main stream. The intra-block kernel is instead streamlined through fusion: the merging of the AttnRes output with its partial-sum update, together with the subsequent RMSNorm, is fused into the preceding TP all-reduce, eliminating a dedicated kernel for the intra-block phase. Together, these optimizations hide the latency of the inter-block pass and reduce the memory traffic of the intra-block phase.

**Stable LatentMoE.** Stable LatentMoE increases both the total number of experts and the number of activated experts per token. The resulting growth in both the expert space and the per-token expert count raises scheduling and coordination overheads, making it difficult for conventional MoE kernels to sustain high hardware utilization. These challenges motivate dedicated kernel optimizations for this module.

To mitigate the overhead of the latent GEMMs, we adopt three optimizations. First, we fuse the latent down-projection with the MoE router into a single GEMM. Second, we shard latent weight matrices across ranks and fuse the output all-gather into the GEMM epilogue using multimem store instructions. Finally, we overlap the resulting communication with other operators, such as the shared-expert computation. Together, these optimizations eliminate redundant weight traffic and duplicated computation, while hiding the communication latency behind computation.

For routed experts, at small batch sizes, the group GEMMs reduce to memory-bound streaming of weight matrices -a regime for which conventional tile-centric kernels are poorly suited due to their compute-oriented design and preprocessing overheads. We instead build the MoE decoding kernel upon the token-centric design of WarpDecode [Bet26], in which each warp is responsible for one output neuron and streams the associated weights directly from memory. To further increase parallelism, we subdivide each warp into finer-grained lane teams, each processing a disjoint subset of experts, followed by a warp-wide reduction of the partial results. In addition, the weight layout is permuted offline at a one-time preprocessing cost, substantially reducing the runtime dequantization overhead.

#### 5.4.3 Fleet-Level Scheduling

Beyond a single serving instance, the challenge shifts from per-request efficiency to predictability: a prefix-cache miss costs orders of magnitude more than a hit, and a burst of million-token requests can starve short ones. We propose two fleet-level scheduling policies to address this: cache-aware affinity scheduling routes each session to the cluster holding its prefix cache while bounding the cost of cluster failures, and budget-based admission control grants each request class its own resource budget so that bursty long-context traffic cannot degrade system-wide SLOs.

**Cache-aware affinity scheduling.** At 1M context, a typical coding input carries a prefix of 400K tokens but requires a prefill increment of only 4K tokens, so a prefix-cache hit avoids re-prefilling the entire prefix and is orders of magnitude cheaper than a miss. We therefore route each request to the cluster that holds its prefix cache, as moving the cache to another cluster would require transferring it over inter-cluster links far slower than the intra-cluster fabric. This cache-aware affinity, however, binds each session to a single cluster, whose failure would interrupt all sessions bound to it. Consistent hashing therefore pins each session to two clusters, a primary that serves its traffic and a pre-assigned secondary that takes over when the primary fails. The secondary holds none of the session's prefix cache and must re-prefill it upon failover. Since consistent hashing distributes the secondary assignments of different sessions uniformly across the fleet, this re-prefill work is divided among many clusters rather than concentrated on one. Cache locality is thus preserved in the common case, while the impact of any single cluster failure remains bounded.

#### Budget-based admission control

Production traffic mixes short requests under 2K tokens with ultra-long requests up to 1M tokens, so the per-request cost spans roughly three orders of magnitude and the total load imposed by any fixed number of requests is highly unpredictable. Capacity planning, queueing models, and rate-limiting quotas based on the "average request" all break down under this variance. In a typical failure mode, a burst of long-context requests saturates the available compute, and short requests arriving afterwards cannot be scheduled promptly, degrading time to first token (TTFT) across all traffic. We therefore adopt budget-based admission control, allocating separate resource budgets to different request classes so that bursty long-context traffic consumes at most its own share of the capacity and cannot degrade system-wide SLOs experienced by other classes.

## 6 Evaluations

### 6.1 Main Results

#### 6.1.1 Benchmarks

We evaluate Kimi K3 on a comprehensive benchmark suite organized along four broad capability axes:

- **Reasoning & Knowledge:** GPQA Diamond [Rei24], CritPt [Art26], AA-LCR [Art26a], and Humanity's Last Exam (HLE-Full, with and without tools) [Pha25].

- **Coding:** DeepSWE [Ela26], ProgramBench [Pro26], Terminal-Bench 2.1 [Mer26], FrontierSWE [Fu24], SWE-Marathon [Mar26], PostTrainBench [Pos26], MLS-Bench-Lite [Lyu26], and SciCode [Tia24, Art26].

- **Agentic:** BrowseComp [Wei25], DeepSearchQA [Ved25], ResearchRubrics [Sha26], Toolathlon-Verified [Li25b], MCPMark-Verified [Wu25], MCP-Atlas [Ban26a], AutomationBench [She26], JobBench [Li26], GDPval-AA v2 [Pat25], AA-Briefcase [Art26, Age26], Agents' Last Exam (ALE) [Age26a, Sun26a], APEX-Agents [Vid26], OfficeQA Pro [Ops26], SpreadsheetBench 2 [Zhu26], OSWorld-Verified [Xie25] and OSWorld 2.0 [Yua26], SaaS-Bench [Shi26], $\tau^3$-Banking [Ban26, Art26], Harvey Lab-AA [Art26, Har26], CorpFin v2 [Val26], Finance Agent v2 [Fro26], and Legal Research Bench [Val26b].

- **Vision:** WorldVQA [Zho26], OmniDocBench [Ouy25], PerceptionBench [Tea26b], Video-MME [GlmWeb], MMVU [Zha25], and BabyVision [Che26] with Python tool. MMMU-Pro [Yue24], CharXiv (RQ) [Wan24a], Math-Vision [Wan24], and ZeroBenchmain [Rob25], each with and without Python tool augmentation.

#### 6.1.2 Baselines

We benchmark against state-of-the-art proprietary and open-source models. For proprietary models, we compare against Claude Fable 5 [Fab26], GPT-5.6 Sol [Sol26], Claude Opus 4.8 [Opu26], and GPT-5.5 [Ope26]. The results of Claude Fable 5 include fallback behaviors and the results of GPT-5.6 Sol include potential cyberguards. For open-source models, we include GLM-5.2 [Z26]. All models are evaluated at maximum reasoning effort, except GPT-5.5, which uses the "xhigh" setting.

#### 6.1.3 Evaluation Configurations

All Kimi K3 evaluations use reasoning effort max and temperature $=1.0$. For single-step tasks, such as GPQA Diamond, HLE-Full, and vision benchmarks without tools, we set top-$p=0.95$. For agentic tasks, we set top-$p=1.0$. Generally, we recommend using top-$p=0.95$ for reasoning and knowledge tasks, and top-$p=1.0$ for coding and agentic scenarios.

**Coding.** Each model is evaluated under one of three agentic harnesses: Kimi Code [Kim26], Claude Code [Cod26], or Codex [Cod26a]. On DeepSWE, we report results on the v1.1 tasks, with additional reference to the official leaderboard (Kimi K3 attains 67.3 with the mini-SWE-agent harness). On Terminal-Bench 2.1, we report the best score across harnesses for all models. Our SWE-Marathon evaluation is based on an H20-calibrated branch of the official tasks as of July 9, 2026, prior to the final v1.1 release, with Docker images, performance gates, and reference oracles for the GPU tasks recalibrated for H20 but the correctness and anti-cheat validators unchanged; Claude Fable 5 hits fallbacks on 35% of the tasks. For PostTrainBench, we evaluate Kimi K3, Claude Fable 5, and GPT-5.6 Sol using the official Harbor implementation at maximum effort, averaged over three runs on H20 GPUs (instead of H100 in the official setting). FrontierSWE dominance scores are recomputed from raw scores using the official evaluation script as of July 16, 2026.

**Agentic.** For OfficeQA Pro, each test case provides the agent with the entire PDF corpus rendered as images, with no machine-readable text available. MCP-Atlas is evaluated on the 500-task public subset with a 100-turn limit, using Gemini 3.1 Pro as the judge. AutomationBench is evaluated on the 600-task public subset. For BrowseComp we adopt a context-compaction strategy triggered at 300K tokens; evaluated with the full 1M-token context window and no context management, Kimi K3 achieves 90.4%.

**Vision.** Scores are averaged over three runs, except ZeroBench-main, which we run five times following the official setting. MMMU-Pro follows the official protocol, preserving the original input order and prepending images to the text input. For WorldVQA, we observe consistent refusal behavior across models and enforce an answer via prompt engineering.

#### Third-party results

GDPval-AA v2, AA-Briefcase, $\tau^3$-Banking, Harvey Lab-AA, APEX-Agents, SciCode, AA-LCR, and CritPt scores are cited from Artificial Analysis [Art26] as of July 23, 2026. For Harvey Lab-AA, we report the criterion pass rate. CorpFin v2, Finance Agent v2, and Legal Research Bench scores are cited from Vals AI [Val26c]. Agents' Last Exam scores are cited from the official leaderboard [Age26a] as of July 23, 2026; we report the leaderboard's primary pass-rate metric. On the leaderboard, each model is paired with a specific harness: Kimi K3 with Kimi Code; GPT-5.6 Sol and GPT-5.5 with Codex; and Claude Fable 5, Claude Opus 4.8, and GLM-5.2 with Claude Code. Toolathlon-verified and JobBench scores are cited from their official leaderboards [The26, Job26] as of July 24, 2026.

#### 6.1.4 Results

![Public benchmark comparison](../../papers/kimi-k3/table-02.png)

**Table 2.** Performance comparison of Kimi K3 against proprietary and open-weight models. Bold denotes the best result for each benchmark and underline denotes the second best.

Table 2 provides a comprehensive comparison of Kimi K3 against both proprietary and open-source baselines. Overall, Kimi K3 closely trails the strongest proprietary models, Claude Fable 5 and GPT-5.6 Sol, while consistently outperforming Claude Opus 4.8, GPT-5.5, and GLM-5.2 across the benchmark suite. We highlight key observations across core capability domains below:

**Reasoning & Knowledge.** On graduate-level reasoning, Kimi K3 is competitive with the frontier, scoring 93.5% on GPQA Diamond. However, a gap remains on research-level tasks: on HLE-Full it trails Claude Fable 5 and GPT-5.6 Sol both with and without tools, at 56.0% and 43.5% respectively; and on CritPt it scores 23.4%, lagging behind Claude Fable 5, GPT-5.6 Sol, and GPT-5.5, indicating that research-level reasoning remains a key direction for improvement.

### 6.2 Internal Evaluation

#### 6.2.1 Capability Evaluation

Beyond the public benchmark suite, we maintain a collection of in-house benchmarks that target capability areas public evaluations do not adequately cover, giving a more comprehensive measure of model and agent capabilities. These benchmarks are refreshed and expanded frequently, so that they can closely track the model's evolving failure modes and directly guide data and training iterations. They broadly fall into three categories: coding capability and experience, general agent experience, and conversational experience. Table 3 reports the results across these benchmarks.

![In-house benchmark results](../../papers/kimi-k3/table-03.png)

**Table 3.** Results on the in-house benchmark suite. Bold denotes the best reported result per benchmark; a dash denotes scores not included in the report.

#### Coding Capability and Experience

- **Kimi Code Bench 2.0 (KCB 2.0):** evaluates code agents on realistic, end-to-end software engineering tasks across a broad range of programming languages and production-oriented technology stacks.

- **Kimi Webdev Bench:** evaluates models on challenging web development prompts drawn from real usage scenarios, with outputs compared through blind expert judgment, with results available in Table 4.

![Kimi Webdev Bench results](../../papers/kimi-k3/table-04.png)

**Table 4.** Results on Kimi Webdev Bench: Kimi K3 against Claude Opus 4.8 under blind expert judging of code quality, feature completeness, visual fidelity, and interaction experience.

- **Coding Experience:** evaluates the practical experience of working with the model as a coding agent in real development workflows.

#### General Agent Experience

- **24/7 ClawBench 2.0:** simulates always-on assistant work, in which tasks span multiple days, events arrive concurrently, and interruptions are routine.

- **Multi-Agent Infra for Routing and Assignment (MIRA) Bench:** evaluates long-chain, multi-role, multi-system enterprise collaboration tasks, assessing whether agents can carry out end-to-end work and judge when to organize or delegate to subagents.

- **Kimi Autonomous Execution Tasks (KAET):** evaluates long-horizon autonomous execution on tasks simulating real user requests and enterprise system operations.

- **Context Learning and Instruction Following (CLIF) Bench:** targets in-context learning, requiring models to learn from a provided context while following instructions that interleave multiple complex skills.

- **Agentic Vision Bench:** evaluates whether agents notice and correctly use key visual facts during task execution.

- **Swarm Bench:** evaluates models' ability to orchestrate agent swarms [Kim26a] on complex tasks that benefit from coordinated decomposition and parallel execution.

- **Online Experience:** mirrors the distribution of real online agent usage, measuring performance on the deliverable file types most frequently requested by users.

- **Deep Research Bench:** evaluates models on deep-research-style queries curated by domain experts and graded with expert-aligned rubrics.

- **Finance Bench:** evaluates models on realistic financial work that requires end-to-end execution of complete workflows, from source materials to reviewable deliverables.

- **Knowledge Work Vision (KWV) Bench:** evaluates atomic visual capabilities extracted from tasks distilled from real knowledge-work scenarios.

- **DECK Bench:** measures the capability to produce high-quality presentation decks from task descriptions drawn from real usage scenarios.

- **Agent Behavior Bench:** extends agent evaluation from outcome correctness to process quality, scoring tool-use behavior, efficiency, and discipline alongside task completion.

#### Conversational Experience

- **Faithfulness:** measures factual hallucination rates in model responses, with each response verified by a fact checker.

- **Chat All-in-One Bench:** measures conversational experience at every stage of product usage, with scenarios designed around real online user needs.

**Evaluation Configurations.** Unless a benchmark is split into separate rows by harness, the Harness column in Table 3 reports the harness used for Kimi K3. For other models, Claude models and GLM-5.2 are evaluated with Claude Code, while GPT models are evaluated with Codex. The exceptions are benchmarks where all models use the same specified harness: OpenClaw for 24/7 ClawBench 2.0; MIRA (Multi-Agent Infra for Routing and Assignment), an internal out-of-distribution harness, for MIRA Bench; Kimi Work for Agent Behavior Bench and Chat All-in-One; and Kimi Code for CLIF and Agentic Vision Bench.

**Results.** The in-house suite separates Kimi K3's strengths from its weaknesses more sharply than the public benchmarks. The clearest strengths are orchestration-and research-type agency: Kimi K3 leads Swarm Bench (76.3) and Deep Research Bench (90.0) by clear margins, indicating strong capability in decomposing complex objectives, coordinating parallel work, and producing rubric-satisfying deliverables. Coding is likewise a strength: on Kimi Code Bench 2.0 it trails only Claude Fable 5, and it attains the best score on Coding Experience, suggesting that its practical behavior as a coding agent -communication quality, behavioral appropriateness, and instruction-following stability -is ahead of its raw task scores; on the Kimi Webdev Bench, expert judges prefer it over Claude Opus 4.8 by a +31.0-point overall margin, with the largest gain on 3D/WebGL/Shader tasks. Professional knowledge work has also improved markedly over the previous generation, with Finance Bench essentially tied with GPT-5.6 Sol.

Kimi K3 trails the leaders mainly on Agent Behavior Bench, MIRA Bench, 24/7 ClawBench 2.0, Agentic Vision Bench, and KWV Bench. On the remaining filled suites (KAET, CLIF Bench, Online Experience, DECK Bench, Faithfulness, and Chat All-in-One Bench), Kimi K3 ranks first or a close second.

#### 6.2.2 Cyber Security Evaluation

We evaluate the model's cybersecurity capability along a two-tier progression of increasing operational risk: vulnerability discovery with proof-of-concept development (Tier 1), and end-to-end exploit development (Tier 2). Evaluation targets include recent versions of widely deployed software-operating-system kernel components and open-source projects-as well as our internal infrastructure, including production services and codebases. All tasks run in standard configurations representative of real-world deployments. Frontier models from Anthropic and OpenAI refuse cyberrelated tasks, making a comparable evaluation infeasible; we therefore exclude them from this suite.

#### Vulnerability discovery (Tier 1).

This tier tasks the model with identifying genuine bugs in current codebases-rather than reproducing known vulnerabilities-and demonstrating that they are reproducible. These capabilities are primarily associated with defensive security research.

Across dozens of widely deployed systems spanning operating-system kernels, databases, AI services, web frameworks, blockchain, and VPN software, the model identified hundreds of candidate vulnerabilities. Of the findings that underwent human review, approximately 70% were confirmed as genuine, including 16 previously unknown vulnerabilities across six projects.

Two findings in the Linux kernel illustrate the depth of these results. First, the model identified a remotely triggerable heap out-of-bounds write. The bug was introduced by an incomplete upstream fix and affects all subsequent releases, up to and including the latest upstream code. Security experts confirmed it as a remote denial-of-service primitive. Second, the model identified a Dirty-COW-class vulnerability in the RDMA subsystem: an earlier upstream fix had inadvertently dropped a permission check, enabling kernel-side writes to read-only memory pages. Security experts confirmed it as a deterministic local privilege-escalation primitive.

**Exploit development (Tier 2).** This tier requires the model to convert a vulnerability into a working end-to-end exploit, and is the tier most directly relevant to misuse risk. We evaluate it against GLM-5.2 as the baseline, using an in-house suite of 36 tasks spanning two tracks.

**User-space exploitation (16 tasks).** The model must exploit real CVEs end-to-end in widely deployed user-space software, including PostgreSQL, the XWiki collaboration platform, the Apache HTTP Server, and several contentmanagement systems and other applications. For each task, the model is given full source code and a live instance; targets run in standard configurations without additional hardening.

**Linux kernel exploitation (20 tasks).** Each task provides a reproducible QEMU environment built from a historical kernel CVE, and the model must write a C exploit that escalates privileges from an unprivileged user to root. Mitigations are progressively enabled across difficulty grades.

Every task in the suite is verified solvable by human security experts. We estimate that completing the full suite requires roughly 540 expert-hours, or about 15 hours per task on average.

**Results on the exploit suite.** The model demonstrates meaningful exploit-development capability on this suite, solving 14 of 36 tasks (38.9%) versus 8 of 36 (22.2%) for GLM-5.2. Its successes are unevenly distributed, however: 10 of the 14 come from the user-space track. On the kernel track, neither model solves three-quarters of the tasks.

Since every task is solvable by human experts, the unsolved tasks directly measure the model's remaining gap to human-level capability. Trajectory analysis attributes this gap to four recurring failure modes: (i) difficulty completing the final stage of an exploit chain from primitives already obtained; (ii) poor strategy selection under mitigations, such as persisting with control-flow hijacking when a data-only attack would be simpler and more reliable; (iii) getting trapped in prolonged, unproductive debugging loops; and (iv) insufficient verification of the final deliverable before submission.

**Summary.** The model's cyber capability is strongest at Tier 1 and at user-space exploitation within Tier 2, yet a clear gap to human experts remains. At Tier 1, which is defensive in nature, the model identifies genuine vulnerabilitiesincluding previously unknown ones-and demonstrates their reproducibility. At Tier 2, it completes end-to-end exploits against user-space targets. Against hardened targets, however, completing the full exploit chain remains the bottleneck, and many expert-solvable tasks go unsolved.

An independent joint assessment by the UK AI Security Institute and NIST's Center for AI Standards and Innovation (CAISI) [Pre26] reaches conclusions consistent with ours. Kimi K3 outperforms GLM-5.2 on exploit development (32% vs. 24% on ExploitBench; 17 vs. 11 steps on a 32-step simulated enterprise network that takes a human expert roughly 20 hours), but trails frontier cyber-capable models on end-to-end exploit completion, achieving arbitrary code execution on 0 of 41 tasks.

We regard our evaluation as a lower bound on capability. These results are conditioned on the current model version and evaluation coverage, and we will revisit them at each major model update.

### 6.3 Third-Party Evaluation

Kimi K3 has also been independently evaluated by third-party organizations since its release. Table 5 summarizes the headline results as of July 23, 2026.

![Independent third-party evaluation results](../../papers/kimi-k3/table-05.png)

**Table 5.** Headline independent third-party evaluations of Kimi K3 as of July 23, 2026.

#### Artificial Analysis

Artificial Analysis evaluated Kimi K3 [Art26]. Kimi K3 attains an Intelligence Index v4.1 of 57.1, ranking fourth of 580 models -third if GPT-5.6 Sol effort variants are counted as a single entry -behind Claude Fable 5 (59.9) and GPT-5.6 Sol (58.9), and ahead of all other evaluated models.

**Vals AI.** On Vals AI's GDP-weighted industry benchmark suite [Val26c], Kimi K3 ranks second of 39 models on the Vals Index (74.7%), behind Claude Fable 5 (75.1%) and ahead of GPT-5.6 Sol (73.1%).

**Arena.** On the crowdsourced human-preference arenas [Lea26], Kimi K3 ranks first of 99 models on the WebDev Arena (1,678 Elo, ahead of Claude Fable 5 at 1,634) -the first open model to top this leaderboard -and eighth of 200 on the Text Arena (1,486 Elo). On the Agent Arena, which opened for voting around July 19, Kimi K3 currently ranks fourth of 37 (9.1), behind Claude Fable 5 (12.7), GPT-5.6 Sol (10.1), and Claude Opus 4.8 (9.8).

### 6.4 Cost Efficiency

Beyond scores, we examine inference cost efficiency by comparing score against per-task cost across four suites covering coding and agentic tasks: Kimi Code Bench 2.0, BrowseComp, GDPval-AA v2, and AA-Briefcase. For Kimi Code Bench 2.0, costs are measured internally, with Kimi K3 run via Kimi Code, and all other models via Claude Code.

For BrowseComp, the cost of Kimi K3 is measured from our own runs, while the costs of Claude and GPT are cited from published charts [Sol26, Son26, Son26a]. For GDPval-AA v2 and AA-Briefcase, costs are cited from Artificial Analysis's pay-per-token API pricing as of July 23, 2026 [Art26].

On Kimi Code Bench 2.0, Kimi K3 is 4.0 points behind Claude Fable 5 at 38% of its cost, and at high effort it already matches Claude Opus 4.8's maximum-effort score at roughly one third of the cost. On BrowseComp, Kimi K3 attains the best score (91.2%) at \$2.03 per task, half the cost of GPT-5.6 Sol (90.4%) and an order of magnitude cheaper than the Claude models at their maximum effort. On GDPval-AA v2, Kimi K3 is within 50 Elo of GPT-5.6 Sol at 13% lower cost, and $2.6\times$ cheaper than Claude Fable 5. On AA-Briefcase, it delivers the second-best score behind Claude Fable 5, at roughly half of the latter's cost. Figure 13 summarizes the comparison. Overall, Kimi K3 sits on or near the cost-efficiency frontier across all four suites, delivering near-top scores at a fraction of the cost of Claude Fable 5 in particular.

![Score versus per-task inference cost](../../papers/kimi-k3/figure-13.png)

**Figure 13.** Score versus per-task inference cost on Kimi Code Bench 2.0, BrowseComp, GDPval-AA v2, and AA-Briefcase. Kimi K3 is marked with a star.

## 7 Case Studies

In this section, we present representative cases that demonstrate Kimi K3's capabilities across diverse technical tasks.

#### GPU kernel optimization

We tested the models' ability to optimize GPU kernels. Each model works independently in an identically configured sandbox, with a budget of up to 24 hours per task for profiling, rewriting, and benchmarking. The evaluation covers four representative kernels: AttnRes, DeepSeek Sparse Attention (DSA), KDA, and MLA (with head dimension 512), on an NVIDIA Hopper GPU and an alternative-vendor GPGPU. Kimi K3 substantially improved performance across all four kernels, reducing AttnRes latency from 283.6 ms to 114.4 ms, cutting DSA and KDA runtime by 55.1% and 73.6%, respectively, and reaching over half of peak TFLOPS on MLA. Across these tasks, Kimi K3 matched Claude Fable 5 [Fab26] (with fallback) and substantially outperformed Claude Opus 4.8 [Opu26], GPT-5.6 Sol [Sol26], and GPT-5.5 [Ope26]. Figure 14 compares the models' optimization trajectories on AttnRes. Beyond the benchmark, an early Kimi K3 checkpoint was already handling most of our kernel optimization work during late-stage development.

![GPU kernel optimization trajectories](../../papers/kimi-k3/figure-14.png)

**Figure 14.** GPU kernel optimization trajectories for AttnRes.

**GPU compiler development.** Kimi K3 developed MiniTriton ([repository](https://github.com/MoonshotAI/minitriton)), a compact Triton-like [Til19] compiler with a custom tile-level Python frontend and layout system, a lightweight warp-level MLIR [Lat21] annotation and optimization layer, and a Parallel Thread Execution (PTX) code-generation pipeline. Built around the compiler is a dual-mode tensor library with a PyTorch-like [Pas19] high-level interface, whose eager and forward-only compiled paths share the same DSL compiler and runtime. The library further provides reverse-mode autograd, neural-network modules, distributed-training primitives over NCCL [NccWeb], and sparse and visualization primitives. On an NVIDIA L20, MiniTriton outperforms PyTorch eager [Pas19] and torch.compile [Ans24] in geometric mean over its core benchmark suite. Its from-scratch tensor-core matmul path approaches cuBLAS [Cub26] at the largest shapes, reaching about 90% of the measured machine roof, while its DSL-level KDA [Tea25b] prefill kernel outperforms a matched Triton reference by a clear margin. MiniTriton also trains a GPT model end to end with a loss curve closely tracking the PyTorch reference, with full-model gradients differing from torch autograd by no more than torch's own FP32 rounding error, $10^{-4}$, measured against an FP64 reference. Together, these results demonstrate that Kimi K3 can build a coherent end-to-end compiler, from DSL frontend and IR passes to PTX code generation and CUDA runtime, rather than a collection of isolated kernels (Figure 15).

![MiniTriton GPU compiler results](../../papers/kimi-k3/figure-15.png)

**Figure 15.** GPU compiler development with MiniTriton: CUDA-core and tensor-core rooflines, GPT training loss, and two-GPU data-parallel training.

#### Chip design

As an early proof of concept, Kimi K3 designed an inference-chip prototype for a nano model following the same architecture, hybrid KDA and NoPE-MLA attention, Block AttnRes with a block size of two, and sigmoid-based MoE routing with one shared expert, under group-wise INT4 weight quantization (group size 128). In a single 48-hour autonomous run with Kimi Code, Kimi K3 built, optimized, and verified the chip using open-source EDA tools with the Nangate45 standard-cell library [Nan10]. Within the $4\,\mathrm{mm}^2$ analytical area budget, the design closes timing at 100 MHz and achieves an RTL-simulated decode throughput of over 8,700 tokens/s, integrating 1.46M standard cells, 0.277 MiB of SRAM, and an INT4 MAC array with fused dequantization. The RTL code is available on GitHub ([repository](https://github.com/MoonshotAI/nano-kpu)).

#### Coding for research

To reproduce the I-Love-Q universal relations in computational astrophysics, Kimi K3 reviewed more than 20 papers and cross-validated their results, implemented the full numerical pipeline, evaluated over 300 equations of state, identified inconsistencies in published formulas, wrote more than 3,000 lines of Python, and produced an interactive HTML dashboard -in about two hours, versus a typical one to two weeks for an experienced researcher.

#### Knowledge work

In Kimi Work, Kimi K3 produced an interactive research website covering 42 years of the AI ASIC industry. The model completed more than 120 rounds of iterative refinement, drawing on a corpus of 87 quarterly reports and 99 original PDFs (more than 11,000 pages) through over 2,800 web searches and over 1,100 terminal queries. In a second case, Kimi K3 analyzed 391 gravitational-wave events in GWTC-5 using more than 20 concurrent subagents, producing seven scientific visualizations, two summary tables, and a literature synthesis of over ten papers.

**Video editing and motion design.** Leveraging its native multimodal architecture, Kimi K3 created a 3Blue1Brown-style motion-graphics explainer of its own architecture, and edited its teaser video from 56 source clips. This involved clip selection, motion-matched cuts, frame-accurate beat synchronization, audio processing, and multiple rounds of revision. Producing a comparable high-density short video would typically take an experienced editor one to two days.

## 8 Conclusion

We present Kimi K3, an open 2.8-trillion-parameter Mixture-of-Experts model with native vision capabilities and a 1-million-token context window, built on Kimi Delta Attention and Attention Residuals. As the world's first open 3T-class model, Kimi K3 delivers frontier-level performance across long-horizon coding, agentic, knowledge, reasoning, and vision tasks. Although gaps to the strongest proprietary models remain, Kimi K3 establishes a new open frontier within everyone's reach. We hope it will empower the broader community in research, deployment, and innovation.

## A Contributions

The complete contributor list is available in the original PDF and is ordered alphabetically by surname.

## B Details of Sigmoid Tanh Unit GLU

The design goal of SiTU-GLU ([§2.3.2](#_2-3-2-sigmoid-tanh-unit-glu)) is to bound the SwiGLU product without discarding the characteristic shape of Swish: an approximately linear response around the origin and a vanishing negative tail. Fig. 4 shows the gate and up branches together with their complete scalar responses.

#### Smoothly capping both branches

SiTU caps the linear factor of Swish as $\beta_1\tanh(\mathbf W_g\mathbf x/\beta_1)$ while retaining the sigmoid factor [Tea26a]. Because the sigmoid already drives the negative gate response toward zero, this change primarily controls large positive activations without removing the negative tail. Kimi K3 applies the same construction to the up branch as $\beta_2\tanh(\mathbf W_u\mathbf x/\beta_2)$, preventing either branch from dominating the product.

**Local and limiting behavior.** For a scalar $z$ near the origin, the scaled tanh satisfies

$$
\beta\tanh(z/\beta)=z+O(z^3/\beta^2).
\tag{18}
$$

SiTU-GLU therefore matches SwiGLU to first order around the origin. It also recovers SwiGLU pointwise as $\beta_1,\beta_2\to\infty$.

**Bounded output.** Since $|\tanh(z)|<1$ and $0<\operatorname{Sigmoid}(z)<1$, every output coordinate satisfies

$$
\|\operatorname{SiTU\text{-}GLU}(\mathbf x)\|_\infty\le\beta_1\beta_2=100.
\tag{19}
$$

Here $\beta_1=4$ and $\beta_2=25$. Unlike hard clamping of gate pre-activations, the smooth cap preserves nonzero gradients away from saturation boundaries, which we find to give better training behavior.

## C Derivation of Quantile Balancing

This appendix derives the Quantile Balancing (QB) updates used in [§2.3](#_2-3-stable-latentmoe) from optimal balanced assignment, following [Su26]; the assignment perspective on expert load balancing goes back to BASE Layers [Lew21] and BIP [Sun25a]. Let $\mathbf s\in\mathbb R^{m\times n}$ collect the router scores of $m$ tokens over $n$ experts, where each token selects exactly $k$ experts and $x_{i,j}\in\{0,1\}$ indicates whether token $i$ is assigned to expert $j$. The maximum-score balanced assignment, in which each expert serves exactly $mk/n$ tokens (assumed integral), is

$$
\max_{\mathbf x;\,x_{i,j}\in\{0,1\}}\sum_{i,j}x_{i,j}s_{i,j}
\quad\text{s.t.}\quad
\sum_jx_{i,j}=k,\qquad\sum_i x_{i,j}=\frac{mk}{n}.
\tag{20}
$$

**Linear relaxation and duality.** Relaxing $x_{i,j}\in\{0,1\}$ to $x_{i,j}\in[0,1]$ turns Equation 20 into a linear program, whose optimum is integral by the standard integrality of the bipartite $b$-matching polytope; the relaxation is therefore exact. Introducing free multipliers $\alpha_i$ and $\beta_j$ for the token- and expert-side equality constraints, respectively, the relaxed problem can be written in max-min form as

$$
\max_{\mathbf x;\,x_{i,j}\in[0,1]}\min_{\boldsymbol\alpha,\boldsymbol\beta}
\sum_{i,j}x_{i,j}s_{i,j}
-\sum_i\alpha_i\left(\sum_jx_{i,j}-k\right)
-\sum_j\beta_j\left(\sum_i x_{i,j}-\frac{mk}{n}\right).
\tag{21}
$$

The objective is linear in each of $\mathbf x$, $\boldsymbol\alpha$, and $\boldsymbol\beta$, and the feasible sets are convex, so the minimax theorem allows exchanging the order of optimization:

$$
\min_{\boldsymbol\alpha,\boldsymbol\beta}\max_{\mathbf x;\,x_{i,j}\in[0,1]}
\sum_{i,j}x_{i,j}(s_{i,j}-\alpha_i-\beta_j)
+k\sum_i\alpha_i+\frac{mk}{n}\sum_j\beta_j.
\tag{22}
$$

The inner maximum is separable over entries, with $x_{i,j}^*=1$ if $s_{i,j}-\alpha_i-\beta_j>0$ and $x_{i,j}^*=0$ if $s_{i,j}-\alpha_i-\beta_j<0$; the tie case has measure zero in practice. Substituting $\mathbf x^*$ gives the convex dual objective

$$
\min_{\boldsymbol\alpha,\boldsymbol\beta}\mathcal L(\boldsymbol\alpha,\boldsymbol\beta):=
\sum_{i,j}\max(0,s_{i,j}-\alpha_i-\beta_j)
+k\sum_i\alpha_i+\frac{mk}{n}\sum_j\beta_j.
\tag{23}
$$

```pseudocode:line-numbers title="Algorithm 1: The alternating QB solver."
Input: score matrix s in R^(m x n)
Output: assignment x in {0,1}^(m x n)

Initialize beta = 0
for t = 1, 2, ..., T:
  alpha <- desc_sort(s - beta, axis=1)[:, k:k+1]
  beta  <- desc_sort(s - alpha, axis=0)[mk/n:mk/n+1]
return x, where x[i,j] = 1 iff j is in argtop_k(s[i] - beta)
```

**Exact coordinate minimization.** We minimize Equation 23 by alternately solving for $\boldsymbol\alpha$ with $\boldsymbol\beta$ fixed and vice versa; each subproblem admits a closed-form exact solution. With $\boldsymbol\beta$ fixed, the problem decouples over tokens, and for token $i$ we solve

$$
\min_\alpha k\alpha+\sum_j\max(0,s_{i,j}-\beta_j-\alpha).
\tag{24}
$$

This objective is piecewise linear in $\alpha$ with slope $k$ minus the number of margins $s_{i,j}-\beta_j$ exceeding $\alpha$; it is therefore minimized exactly when $k$ margins lie above $\alpha$, i.e., for any $\alpha_i^*$ between the $k$-th and $(k+1)$-th largest entries of $\mathbf s_i-\boldsymbol\beta$. By convention we take the $(k+1)$-th largest entry, equivalently the $(1-k/n)$-th quantile:

$$
\alpha_i^*=\operatorname{quantile}_{1-k/n}(\mathbf s_i-\boldsymbol\beta).
\tag{25}
$$

Symmetrically, with $\boldsymbol\alpha$ fixed, expert $j$ solves $\min_\beta \frac{mk}{n}\beta+\sum_i\max(0,s_{i,j}-\alpha_i-\beta)$, whose minimizer is the $(mk/n+1)$-th largest entry of $\mathbf s_{:,j}-\boldsymbol\alpha$, again the $(1-k/n)$-th quantile:

$$
\beta_j^*=\operatorname{quantile}_{1-k/n}(\mathbf s_{:,j}-\boldsymbol\alpha).
\tag{26}
$$

Both updates are thus the same quantile along the token and expert axes, respectively, which gives the method its name. Fig. 5 illustrates the expert-side update as equalizing the accepted upper tail of each expert's margin distribution, and Alg. 1 summarizes the resulting alternating solver.

**From assignment to routing.** At the optimum of Equation 23, $x_{i,j}^*=1$ if and only if $s_{i,j}-\alpha_i^*-\beta_j^*>0$; combined with the token constraint $\sum_jx_{i,j}^*=k$, the selected experts are exactly the Top-$k$ entries of $\mathbf s_i-\boldsymbol\beta^*$. Routing therefore requires only the expert thresholds $\boldsymbol\beta\in\mathbb R^n$, equivalently the bias $\mathbf b=-\boldsymbol\beta$ of Equation 13, while the token thresholds $\boldsymbol\alpha\in\mathbb R^m$ are intermediate variables tied to the dynamic training batch and are discarded. This asymmetry preserves train-inference consistency: at deployment, routing is a fixed Top-$k$ selection with a frozen bias, and no quantile computation is needed.

**Relation to sign-based loss-free updates.** The expert-side subproblem underlying Equation 26 has (sub)gradient

$$
\frac{\partial\mathcal L}{\partial\beta_j}=\frac{mk}{n}
-\sum_{i=1}^{m}\chi(s_{i,j}-\alpha_i-\beta_j>0).
\tag{27}
$$

i.e., the target load minus the observed load of expert $j$. A SignSGD step on this objective recovers the fixed-step sign update of auxiliary-loss-free balancing [Dee24a], up to the sign convention $\mathbf b=-\boldsymbol\beta$: the sign update retains only the direction of the load error in Equation 27, whereas QB jumps directly to the exact coordinate minimizer of the same dual objective. This view explains both why QB requires no learning-rate-like hyperparameter and why it equilibrates within a few update steps even for nearly $10^3$ experts. QB is likewise related to BIP [Sun25a], which solves the same assignment with inequality constraints $\sum_jx_{i,j}\le k$ and $\sum_i x_{i,j}\le mk/n$; the induced non-negativity constraints on $\boldsymbol\alpha$ and $\boldsymbol\beta$ add a $\max(0,\cdot)$ clipping to both updates, which can only suppress over-selected experts without promoting under-selected ones, and markedly slows equilibration in our experiments. Finally, the resulting fixed-Top-$k$ routing is related to expert-specific threshold routing but differs from Expert Threshold routing, which maintains EMA thresholds and permits a variable number of selected experts per token [Sun26].

## D Histogram-Based Quantile Estimation

The QB update of Equation 14 asks for a quantile taken over the whole training step: for each of the $n$ experts, the $(1-k/n)$-th quantile of the margins $s_{i,j}-\alpha_i$, where the token count $m$ spans millions of tokens sharded across data-parallel ranks and gradient-accumulation steps. Gathering $O(mn)$ margins for an exact quantile is impractical inside the training loop.

The key observation is that the update never needs the margins themselves, only their per-expert distribution, which a histogram summarizes at fixed cost. Kimi K3 therefore maintains a binned histogram per expert and reads the quantile from it. Concretely, we histogram the required bias $r_{i,j}:=\alpha_i-s_{i,j}$, the bias that would place expert $j$ exactly at token $i$'s cutoff; negating the margins reverses their order, so the QB target $b_j$ of Equation 14 is exactly the $(k/n)$-quantile of $\mathbf r_{:,j}$.

#### Binning range

The first question is which interval to bin over, and here the required bias helps: its range is bounded by the current bias itself. Router scores are sigmoid outputs, so $s_{i,j}\in(0,1)$, and the cutoff $\alpha_i$ is itself the biased score $s_{i,j'}+b_{j'}$ of some expert $j'$, so it lies in $(b_{\min},1+b_{\max})$, with $b_{\min}$ and $b_{\max}$ the extremes of the current bias. Every $r_{i,j}$ therefore falls in $[b_{\min}-1,b_{\max}+1]$. We partition this interval into $B$ uniform bins and recompute the range every step, so the bin width $w=(b_{\max}-b_{\min}+2)/B$ stays adapted to the bias as it spreads to correct imbalance.

#### Accumulation and recovery

The rest of the procedure follows the structure of a training step. During each forward pass, every rank scatter-adds its local $r_{i,j}$ values into a per-expert count matrix $\mathbf H\in\mathbb N^{n\times B}$, accumulating over all micro-batches with no communication. At the end of the step, a single all-reduce sums the local counts into the global histogram, and every rank recovers the quantile from the same pooled counts. Each expert's histogram counts every token once, so the target rank is exactly $q=mk/n$ ([§2.3.3](#_2-3-3-quantile-balancing)), now taken over the full step: we select the first bin whose cumulative count reaches $\lceil q\rceil$ and interpolate linearly within it. If bin $\ell_j$ is selected, with cumulative count $c_j$ before it and $h_j$ counts inside it, then $b_j=b_{\min}-1+(\ell_j+\operatorname{clip}((q-c_j)/h_j,0,1))w$, and the resulting biases are mean-centered as in Equation 14.

**Properties.** Three properties make this estimator practical at scale. First, it is accurate: the cumulative counts are exact at bin edges, so the true quantile and its estimate lie in the same bin and the error is bounded by the bin width $w$; with $B=1000$ this is at most a few $10^{-3}$, and we observe no measurable residual load imbalance. Second, it is cheap: the only communication is one integer all-reduce of $nB$ values per layer per step, independent of $m$, which in our configuration is below 1% of the cost of exchanging the raw margins over a process group every micro-batch, the natural alternative. Third, it estimates the right quantity: because counts are additive, the global histogram is exactly invariant to how tokens are partitioned across ranks or accumulation steps, and the estimate is the quantile of the pooled global batch rather than an average of per-rank quantiles, which generally differs. As a further refinement, maintaining an exponential moving average of the estimated quantiles across steps reduces batch-to-batch sampling noise and can improve load balance still further.

## E MoonEP General Upper Bound Proof

Let $m_r(P)$ denote the number of redundant experts placed on rank $r$ under plan $P$. For a router output $I$, the planning objective is to minimize the maximum number of redundant experts on any rank, i.e., $M(I)=\min_P\max_r\{m_r(P)\}$. We prove that $M(I)\le E/R$ always holds (Theorem 1) and that this bound is essentially tight: there exist router outputs for which $M=\lceil E(R-1)/R^2\rceil\approx E/R$ (Theorem 2).

#### Proof of Theorem 1 (General Upper Bound)

The goal is to prove that $M(I)\le E/R$ holds for any router output $I$. The key lemma is that there exists a plan $P^*$ such that every EP rank receives exactly the same number of tokens, $S\times K$, and the remote tokens of each rank come from only one other EP rank. Initially, every rank holds only local tokens, and ranks are classified as underloaded or overloaded. We repeatedly pick one of each and migrate tokens from the overloaded rank to fill the underloaded rank exactly to $S\times K$. Each fill makes one underloaded rank balanced and it never changes afterward, so the process terminates after at most $R-1$ fills. Each rank is filled at most once, so its remote tokens come from a single rank. If the remote tokens of rank $r$ come from rank $s$, they belong to at most $E/R$ local experts on rank $s$, hence $m_r(P^*)\le E/R$, and therefore

$$
M(I)=\min_P\max_r\{m_r(P)\}
\le\max_r\{m_r(P^*)\}\le\frac ER.
\tag{28}
$$

#### Proof of Theorem 2 (Tightness of the Upper Bound)

Construct a router output $I^*$ as follows: the experts on EP rank 0 receive no tokens, while all experts on the other $R-1$ ranks share all tokens evenly. Then all $S\times K\times R$ tokens are evenly divided among $E(R-1)/R$ experts, so each expert receives $SKR^2/[E(R-1)]$ tokens. Under any plan $P$, rank 0 must receive $S\times K$ tokens, all of which are remote, and these tokens involve at least $E(R-1)/R^2$ distinct experts. Taking the ceiling, rank 0 requires at least $\lceil E(R-1)/R^2\rceil$ redundant experts, hence $M(I^*)\ge\lceil E(R-1)/R^2\rceil$. Conversely, the filling procedure from Theorem 1, with expert-wise migration prioritized, keeps every rank within this value, so equality holds. Since $E(R-1)/R^2\approx E/R$ when $R$ is large, the upper bound in Theorem 1 is essentially tight.

## F Chat Template

![Kimi K3 chat template structure](../../papers/kimi-k3/figure-16.png)

**Figure 16.** Structure of the Kimi K3 chat template: context layout, assistant-message channels, and indexed parallel tool calls.

The Kimi K3 chat template is redesigned around three goals. The first is extensibility: new capabilities should be introduced through backward-compatible message formats rather than template revisions, so that a single template serves the entire model generation. The second is a low alignment tax: the format should be learnable with minimal supervised data, supporting a pipeline in which a lightly fine-tuned pre-trained model can proceed directly to reinforcement learning. The third is decoding friendliness: the structure should admit simple encoders, streaming parsers, and grammar-constrained enforcers. To these ends, the template adopts XTML (eXtensible Token Markup Language), an XML-like markup in which the angle-bracket syntax is replaced by three reserved special tokens:

#### Messages and zones

The top-level unit of the context is the message, and messages fall into two categories by origin (Fig. 16a). Input messages serialize the messages field of the request, covering the familiar system, user, assistant, and tool roles. Option messages translate request options into instructions that the model reads in context, and their placement reflects their scope. Global options-the tool declaration (type="tool-declare") and the reasoning-effort setting-appear before all input messages: they govern the whole session and rarely change, so modifying them invalidates the KV cache anyway. One-shot options (tool_choice, response_format) are appended after the input messages, so that per-request changes leave the history KV cache intact. A third kind, the input option message, is interleaved with input messages to supplement or override a global option mid-session. This mechanism supports dynamically loaded tools: tools retrieved or loaded during a conversation are announced through an additional tool-declare message, after which the model's available toolset expands without rebuilding the preceding context.

#### Channels

The body of an assistant message is organized into channels, a concept inspired by OpenAI's Harmony response format [Ope25a]: think carries the reasoning trace, response the user-visible answer, and tools the tool calls (Fig. 16b). The two generation modes are selected purely through the generation prefix-[open]think[sep] for thinking mode and [open]response[sep] for instruct mode-rather than through separate templates. Kimi K3 supports only preserved thinking: in thinking mode, the think channel is always retained in the history-kept even when its content is empty-so that the model observes a consistent message structure across turns; in instruct mode, historical messages contain only the response and tools channels.

**Tool calling.** Within the tools channel, each call carries tool and index attributes; the index numbers parallel calls within a message, and each tool-result message repeats the same tool/index pair and follows the order of its call, so that results are unambiguously associated with calls. Arguments are typed: string arguments appear as raw text, while values of other JSON types are compactly serialized. Free-form text such as code is therefore a first-class citizen rather than an escaped JSON string. A pure-JSON fallback block covers inputs whose arguments cannot be decomposed into typed argument blocks; it occurs only in input tokens, never in model outputs, and its loss is masked during training.

#### Reasoning effort and options

Reasoning effort is exposed as a global option message of type thinking-effort, inserted after the tool declaration and before the input messages. Instead of modifying the generation prefix or exposing a token budget, the message states the requested level in natural language and acts as a generation-constraint instruction. The schema reserves four levels (low, medium, high, and max), of which Kimi K3 supports a subset. This representation decouples the effort interface from the template syntax, and it aligns directly with the effort-conditioned training described in [§4.1.1](#_4-1-1-supervised-fine-tuning) and [§4.1.2](#_4-1-2-reinforcement-learning). More broadly, this is the common implementation of all option messages: tool_choice, response_format, and thinking-effort are each translated into a short natural-language instruction placed in context, rather than into dedicated special syntax. Because the pre-trained model already follows such instructions well, new options can be introduced with little or no additional training-a direct embodiment of the low-alignment-tax design principle stated above.
