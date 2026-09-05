---
title: 'GQA'
createTime: 2026/09/05 14:30:54
permalink: /en/papers/gqa/
---

> [Joshua Ainslie](https://dblp.org/pid/263/3363) [+1], [James Lee-Thorp](https://dblp.org/pid/292/3901) [+1], [Michiel de Jong](https://dblp.org/pid/223/0153) [+1] [+2], [Yury Zemlyanskiy](https://dblp.org/pid/225/5302), [Federico Lebrón](https://dblp.org/pid/347/9919), and [Sumit Sanghai](https://dblp.org/pid/263/3559). First submitted to arXiv on May 22, 2023; current version v3. Published in the *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, pages 4895-4901, December 2023. [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245). <a href="/paper/gqa.pdf" target="_blank" rel="noopener noreferrer">Original PDF</a>. [EMNLP 2023](https://aclanthology.org/2023.emnlp-main.298/). [DOI](https://doi.org/10.18653/v1/2023.emnlp-main.298). [TeX source](https://export.arxiv.org/e-print/2305.13245). The original PDF remains authoritative for the exact print layout and bibliography.

## Abstract

Multi-query attention (MQA), which only uses a single key-value head, drastically speeds up decoder inference. However, MQA can lead to quality degradation, and moreover it may not be desirable to train a separate model just for faster inference. We (1) propose a recipe for uptraining existing multi-head language model checkpoints into models with MQA using 5% of original pre-training compute, and (2) introduce grouped-query attention (GQA), a generalization of multi-query attention which uses an intermediate (more than one, less than number of query heads) number of key-value heads. We show that uptrained GQA achieves quality close to multi-head attention with comparable speed to MQA.

<span id="section-1"></span>

## 1 Introduction

Autoregressive decoder inference is a severe bottleneck for Transformer models due to the memory bandwidth overhead from loading decoder weights and all attention keys and values at every decoding step [Sha19, Pop22, Dej22]. The memory bandwidth from loading keys and values can be sharply reduced through *multi-query attention* [Sha19], which uses multiple query heads but single key and value heads.

However, multi-query attention (MQA) can lead to quality degradation and training instability, and it may not be feasible to train separate models optimized for quality and inference. Moreover, while some language models already use multi-query attention, such as PaLM [Cho22b], many do not, including publicly available language models such as T5 [Raf20b] and LLaMA [Tou23].

This work contains two contributions for faster inference with large language models. First, we show that language model checkpoints with multi-head attention (MHA) can be *uptrained* [Kom22a] to use MQA with a small fraction of original training compute. This presents a cost-effective method to obtain fast multi-query as well as high-quality MHA checkpoints.

Second, we propose grouped-query attention (GQA), an interpolation between multi-head and multi-query attention with single key and value heads *per subgroup of query heads*. We show that uptrained GQA achieves quality close to multi-head attention while being almost as fast as multi-query attention.

<span id="section-2"></span>

## 2 Method

<span id="section-2-1"></span>

### 2.1 Uptraining

Generating a multi-query model from a multi-head model takes place in two steps: first, converting the checkpoint, and second, additional pre-training to allow the model to adapt to its new structure. [Figure 1](#figure-01) shows the process for converting a multi-head checkpoint into a multi-query checkpoint. The projection matrices for key and value heads are mean pooled into single projection matrices, which we find works better than selecting a single key and value head or randomly initializing new key and value heads from scratch.

<span id="figure-01"></span>

![Diagram showing key projection matrices from multiple heads being mean pooled into one MQA key projection matrix](../../papers/gqa/figure-01.png)

**Figure 1.** Overview of conversion from multi-head to multi-query attention. Key and value projection matrices from all heads are mean pooled into a single head.

The converted checkpoint is then pre-trained for a small proportion $\alpha$ of its original training steps on the same pre-training recipe.

<span id="section-2-2"></span>

### 2.2 Grouped-query attention

<span id="figure-02"></span>

![Comparison of multi-head, grouped-query, and multi-query attention head layouts](../../papers/gqa/figure-02.png)

**Figure 2.** Overview of grouped-query method. Multi-head attention has H query, key, and value heads. Multi-query attention shares single key and value heads across all query heads. Grouped-query attention instead shares single key and value heads for each *group* of query heads, interpolating between multi-head and multi-query attention.

Grouped-query attention divides query heads into $G$ *groups*, each of which shares a single key head and value head. GQA-G refers to grouped-query with $G$ groups. GQA-$1$, with a single group and therefore single key and value head, is equivalent to MQA, while GQA-H, with groups equal to number of heads, is equivalent to MHA. [Figure 2](#figure-02) shows a comparison of grouped-query attention and multi-head/multi-query attention. When converting a multi-head checkpoint to a GQA checkpoint, we construct each group key and value head by mean-pooling all the original heads within that group.

An intermediate number of groups leads to an interpolated model that is higher quality than MQA but faster than MHA, and, as we will show, represents a favorable trade-off. Going from MHA to MQA reduces $H$ key and value heads to a single key and value head, reducing the size of the key-value cache and therefore amount of data that needs to be loaded by a factor of $H$. However, larger models generally scale the number of heads, such that multi-query attention represents a more aggressive cut in both memory bandwidth and capacity. GQA lets us keep the same proportional decrease in bandwidth and capacity as model size increases.

Moreover, larger models suffer relatively less from memory bandwidth overhead from attention, as the KV-cache scales with model dimension while model FLOPs and parameters scale with the square of model dimension. Finally, standard sharding for large models replicates the single key and value head by the number of model partitions [Pop22]; GQA removes the waste from such partitioning. Therefore, we expect GQA to present a particularly good trade-off for larger models.

We note that GQA is not applied to the encoder self-attention layers; encoder representations are computed in parallel, and memory bandwidth is therefore generally not the primary bottleneck.

<span id="section-3"></span>

## 3 Experiments

<span id="section-3-1"></span>

### 3.1 Experimental setup

**Configurations.** All models are based on the T5.1.1 architecture [Raf20b], implemented with JAX [Bra18], Flax [Hee20], and Flaxformer [+3]. For our main experiments we consider T5 Large and XXL with multi-head attention, as well as uptrained versions of T5 XXL with multi-query and grouped-query attention. We use the Adafactor optimizer with the same hyperparameters and learning rate schedule as T5 [Raf20b]. We apply MQA and GQA to decoder self-attention and cross-attention, but not encoder self-attention.

**Uptraining.** Uptrained models are initialized from public T5.1.1 checkpoints. The key and value heads are mean-pooled to the appropriate MQA or GQA structure, and then pre-trained for a further $\alpha$ proportion of original pre-training steps with the original pre-training setup and dataset from [Raf20b]. For $\alpha=0.05$, training took approximately 600 TPUv3 chip-days.

**Data.** We evaluate on summarization datasets CNN/Daily Mail [Nal16], arXiv and PubMed [Coh18], MediaSum [Zhu21a], and Multi-News [Fab19]; translation dataset WMT 2014 English-to-German; and question answering dataset TriviaQA [Jos17]. We do not evaluate on popular classification benchmarks such as GLUE [Wan18d] as autoregressive inference is less applicable for those tasks.

**Fine-tuning.** For fine-tuning, we use a constant learning rate of 0.001, batch size 128, and dropout rate 0.1 for all tasks. CNN/Daily Mail and WMT use input length of 512 and output length 256. Other summarization datasets use input length 2048 and output length 512. Finally, TriviaQA uses input length 2048 and output length 32. We train until convergence and select the checkpoint with the highest dev performance. We use greedy decoding for inference.

**Timing.** We report time per sample per TPUv4 chip, as measured by xprof [Xpr20]. For timing experiments we use 8 TPUs with the largest batch size that fits up to 32 per TPU, and parallelization optimized separately for each model.

<span id="table-01"></span>

![Table comparing inference time and dev-set performance for T5 Large and XXL attention variants](../../papers/gqa/table-01.png)

**Table 1.** Inference time and average dev set performance comparison of T5 Large and XXL models with multi-head attention, and 5% uptrained T5-XXL models with multi-query and grouped-query attention on summarization datasets CNN/Daily Mail, arXiv, PubMed, MediaSum, and MultiNews, translation dataset WMT, and question-answering dataset TriviaQA.

<span id="figure-03"></span>

![Scatter plot of average performance against inference time for MHA, MQA, and GQA models](../../papers/gqa/figure-03.png)

**Figure 3.** **Uptrained MQA yields a favorable tradeoff compared to MHA with higher quality and faster speed than MHA-Large, and GQA achieves even better performance with similar speed gains and comparable quality to MHA-XXL.** Average performance on all tasks as a function of average inference time per sample for T5-Large and T5-XXL with multi-head attention, and 5% uptrained T5-XXL with MQA and GQA-8 attention.

<span id="section-3-2"></span>

### 3.2 Main results

[Figure 3](#figure-03) shows average performance over all datasets as a function of average inference time for MHA T5-Large and T5-XXL, and uptrained MQA and GQA-$8$ XXL models with uptraining proportion $\alpha = 0.05$. We see that a larger uptrained MQA model provides a favorable trade-off relative to MHA models, with higher quality and faster inference than MHA-Large. Moreover, GQA achieves significant additional quality gains, achieving performance close to MHA-XXL with speed close to MQA. [Table 1](#table-01) contains full results for all datasets.

<span id="section-3-3"></span>

### 3.3 Ablations

This section presents experiments to investigate the effect of different modeling choices. We evaluate performance on a representive subsample of tasks: CNN/Daily Mail, (short-form summarization), MultiNews (long-form summarization), and TriviaQA (question-answering).

<span id="figure-04"></span>

![Horizontal bars comparing mean pooling, first-head selection, and random initialization](../../papers/gqa/figure-04.png)

**Figure 4.** Performance comparison of different checkpoint conversion methods for T5-Large uptrained to MQA with proportion $\alpha=0.05$. ‘Mean’ mean-pools key and value heads, ‘First’ selects the first head and ‘Random’ initializes heads from scratch.

**Checkpoint conversion.** [Figure 4](#figure-04) compares the performance of different methods for checkpoint conversion. Mean pooling appears to work best, followed by selecting a single head and then random initialization. Intuitively, results are ordered by the degree to which information is preserved from the pre-trained model.

**Uptraining steps.** [Figure 5](#figure-05) shows how performance varies with uptraining proportion for T5 XXL with MQA and GQA. First, we note that GQA already achieves reasonable performance after conversion while MQA requires uptraining to be useful. Both MQA and GQA gain from 5% uptraining with diminishing returns from 10%.

<span id="figure-05"></span>

![Line chart of MHA, GQA, and MQA performance against uptraining proportion](../../papers/gqa/figure-05.png)

**Figure 5.** Performance as a function of uptraining proportion for T5 XXL models with MQA and GQA-8.

**Number of groups.** [Figure 6](#figure-06) demonstrates the effect of the number of GQA groups on inference speed. For larger models the memory bandwidth overhead from the KV cache is less constraining [Sha19], while the reduction in key-value size is sharper due to the increased number of heads. As a result, increasing the number of groups from MQA only results in modest slowdowns initially, with increasing cost as we move closer to MHA. We selected 8 groups as a favorable middle ground.

<span id="figure-06"></span>

![Line chart of time per sample against the number of GQA groups](../../papers/gqa/figure-06.png)

**Figure 6.** Time per sample for GQA-XXL as a function of the number of GQA groups with input length 2048 and output length 512. Going from 1 (MQA) to 8 groups adds modest inference overhead, with increasing cost to adding more groups.

<span id="section-4"></span>

## 4 Related Work

This work is focused on achieving a better trade-off between decoder quality and inference time through reducing the memory bandwidth overhead [Wil09] from loading keys and values. Shazeer [Sha19] first proposed reducing this overhead through multi-query attention. Follow-up work showed that multi-query attention is especially helpful for long inputs [Pop22, Dej22]. Rabe [Rab23] independently developed GQA with public implementation. Other works have explored grouping attention heads for computational efficiency [Par20b, Luo22, Ni23] without focusing specifically on key-value heads, which determine memory bandwidth overhead.

A number of other methods have been proposed to reduce memory bandwidth overhead from keys and values, as well as parameters. Flash attention [Dao22] structures the attention computation to avoid materializing the quadratic attention scores, reducing memory and speeding up training. Quantization [Det22, Fra22] reduces the size of weights and activations, including keys and values, by lowering precision. Model distillation [Hin15, Gou21] instead reduces model size at a given precision, using data generated from the larger model to finetune the smaller model. Layer-sparse cross-attention [Dej22] eliminates most cross-attention layers which make up the primary expense for longer inputs. Speculative sampling [Che23, Lev23] ameliorates the memory bandwidth bottleneck by proposing multiple tokens with a smaller model which are then scored in parallel by a larger model.

Finally, the uptraining procedure we propose is inspired by Komatsuzaki et al. [Kom22a], which uptrains standard T5 checkpoints into sparsely activated Mixture-of-Experts models.

<span id="section-5"></span>

## 5 Conclusion

Language models are expensive for inference primarily due to the memory bandwidth overhead from loading keys and values. Multi-query attention reduces this overhead at the cost of decreased model capacity and quality. We propose to convert multi-head attention models to multi-query models with a small fraction of original pre-training compute. Moreover, we introduce grouped-query attention, an interpolation of multi-query and multi-head attention that achieves quality close to multi-head at comparable speed to multi-query attention.

<span id="section-6"></span>

## 6 Limitations

This paper focuses on ameliorating the memory bandwidth overhead from loading keys and values. This overhead is most important when generating longer sequences, for which quality is inherently difficult to evaluate. For summarization we employ Rouge score, which we know is a flawed evaluation that does not tell the whole story; for that reason, it is difficult to be certain our trade-offs are correct. Due to limited computation, we also do not compare our XXL GQA model to a comparitive model trained from scratch, so we do not know the relative performance of uptraining vs training from scratch. Finally, we evaluate the impact of uptraining and GQA only on encoder-decoder models. Recently, decoder-only models are extremely popular, and since these models do not have separate self-attention and cross-attention, we expect GQA to have a stronger advantage over MQA.

## Acknowledgements

We thank Santiago Ontañón, Afroz Mohiuddin, William Cohen and others at Google Research for insightful advice and discussion.

<span id="section-7"></span>

## 7 Training Stability

We find that multi-query attention can lead to training instability during fine-tuning, in particular combined with long input tasks. We trained multiple T5-Large models with multi-query attention from scratch. In each case, pre-training suffered from frequent loss spikes and the final models diverged immediately when fine-tuning on long-input tasks. Uptrained multi-query attention models are more stable but still display high variance, so for multi-query models on unstable tasks we report average performance over three fine-tuning runs. Uptrained grouped-query attention models, however, appear to be stable, so we did not investigate futher on the root causes of multi-query instability.

[+1]: Equal contribution.

[+2]: University of Southern California. Work done at Google Research.

[+3]: [https://github.com/google/flaxformer](https://github.com/google/flaxformer)
