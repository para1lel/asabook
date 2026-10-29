---
title: Scaling up GANs for Text-to-Image Synthesis
createTime: 2026/10/29 00:00:00
permalink: /en/papers/gigagan/
pageClass: paper-reading
---

> [Minguk Kang](https://mingukkang.github.io/), [Jun-Yan Zhu](https://www.cs.cmu.edu/~junyanz/), [Richard Zhang](https://richzhang.github.io/), [Jaesik Park](https://jaesik.info/), [Eli Shechtman](https://research.adobe.com/person/eli-shechtman/), [Sylvain Paris](https://www.sylvainparis.com/), and [Taesung Park](https://taesung.me/). First submitted to arXiv on March 9, 2023; current version v2. Published in the [2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition](https://openaccess.thecvf.com/content/CVPR2023/html/Kang_GigaGAN_A_Unified_Approach_to_Generative_Adversarial_Networks_for_Text-to-Image_CVPR_2023_paper.html). [Scaling up GANs for Text-to-Image Synthesis](https://arxiv.org/abs/2303.05511). <a href="/paper/gigagan.pdf" target="_blank" rel="noopener noreferrer">Original PDF</a>. [TeX source](https://export.arxiv.org/e-print/2303.05511v2). The original PDF remains authoritative for the exact print layout and bibliography.

## Abstract

The recent success of text-to-image synthesis has taken the world by storm and captured the general public's imagination. From a technical standpoint, it also marked a drastic change in the favored architecture to design generative image models. GANs used to be the de facto choice, with techniques like StyleGAN. With DALL$\cdot$E 2, autoregressive and diffusion models became the new standard for large-scale generative models overnight. This rapid shift raises a fundamental question: can we scale up GANs to benefit from large datasets like LAION? We find that naively increasing the capacity of the StyleGAN architecture quickly becomes unstable. We introduce GigaGAN, a new GAN architecture that far exceeds this limit, demonstrating GANs as a viable option for text-to-image synthesis. GigaGAN offers three major advantages. First, it is orders of magnitude faster at inference time, taking only 0.13 seconds to synthesize a 512px image. Second, it can synthesize high-resolution images, for example, 16-megapixel images in 3.66 seconds. Finally, GigaGAN supports various latent space editing applications such as latent interpolation, style mixing, and vector arithmetic operations.

<span id="section-1"></span>
## 1 Introduction
<span id="figure-01"></span>

![Figure 1](../../papers/gigagan/figure-01.png)

**Figure 1.** Our model, GigaGAN, shows GAN frameworks can also be scaled up for general text-to-image synthesis tasks, generating a 512px output at an interactive speed of 0.13s, and 4096px at 3.7s. Selected examples at 2K or 4K resolutions are shown. Please zoom in for more details. See Appendix 8 and our website for more uncurated comparisons.
<span id="figure-02"></span>

![Figure 2](../../papers/gigagan/figure-02.png)

**Figure 2.** Our GAN-based upsampler can serve in the upsampling pipeline of many text-to-image models that often generate initial outputs at low resolutions like 64px or 128px. We simulate such usage by applying our text-conditioned 8$\times$ superresolution model on a low-res 128px artwork to obtain the 1K output, using “Portrait of a colored iguana dressed in a hoodie". Then our model can be re-applied to go beyond 4K. We compare our model with the text-conditioned upscaler of Stable Diffusion [Sta22] and unconditional Real-ESRGAN [Wan21i]. Zooming in is recommended for comparison between 1K and 4K.
<span id="figure-03"></span>

![Figure 3](../../papers/gigagan/figure-03.png)

**Figure 3.** Our GAN-based upsampler, similar to [Figure 2](#figure-02), can also be used as an off-the-shelf superresolution model for real images with a large scaling factor by providing an appropriate description of the image. We apply our text-conditioned 8$\times$ superresolution model on a low-res 128px photo to obtain the 1K output, using “A dog sitting in front of a mini tipi tent". Then our model can be re-applied to go beyond 4K. We compare our model with the text-conditioned upscaler of Stable Diffusion [Sta22] and unconditional Real-ESRGAN [Wan21i]. Zooming in is recommended for comparison between 1K and 4K.
Recently released models, such as DALL$\cdot$E 2 [Ram22], Imagen [Sah22a], Parti [Yu22], and Stable Diffusion [Rom22], have ushered in a new era of image generation, achieving unprecedented levels of image quality and model flexibility. The now-dominant paradigms, diffusion models and autoregressive models, both rely on iterative inference. This is a double-edged sword, as iterative methods enable stable training with simple objectives but incur a high computational cost during inference.
Contrast this with Generative Adversarial Networks (GANs) [Goo14a, Rad15, Kar19, Bro19], which generate images through a single forward pass and thus inherently efficient. While such models dominated the previous “era” of generative modeling, scaling them requires careful tuning of the network architectures and training considerations due to instabilities in the training procedure. As such, GANs have excelled at modeling single or multiple object classes, but scaling to complex datasets, much less an open world, has remained challenging. As a result, ultra-large models, data, and compute resources are now dedicated to diffusion and autoregressive models. In this work, we ask – *can GANs continue to be scaled up and potentially benefit from such resources, or have they plateaued? What prevents them from further scaling, and can we overcome these barriers?*
We first experiment with StyleGAN2 [Kar20b] and observe that simply scaling the backbone causes unstable training. We identify several key issues and propose techniques to stabilize the training while increasing the model capacity. First, we effectively scale the generator’s capacity by retaining a bank of filters and taking a sample-specific linear combination. We also adapt several techniques commonly used in the diffusion context and confirm that they bring similar benefits to GANs. For instance, interleaving both self-attention (image-only) and cross-attention (image-text) with the convolutional layers improves performance.
Furthermore, we reintroduce multi-scale training, finding a new scheme that improves image-text alignment and low-frequency details of generated outputs. Multi-scale training allows the GAN-based generator to use parameters in low-resolution blocks more effectively, leading to better image-text alignment and image quality. After careful tuning, we achieve stable and scalable training of a one-billion-parameter GAN (GigaGAN) on large-scale datasets, such as LAION2B-en [Sch22b]. Our results are shown in [Figure 1](#figure-01).
In addition, our method uses a multi-stage approach [Den15, Zha17c]. We first generate at $64\times 64$ and then upsample to $512\times 512$. These two networks are modular and robust enough to be used in a plug-and-play fashion. We show that our text-conditioned GAN-based upsampling network can be used as an efficient, higher-quality upsampler for a base diffusion model such as DALL$\cdot$E 2, despite never having seen diffusion images at training time ([Figure 2](#figure-02)).
Together, these advances enable our GigaGAN to go far beyond previous GANs: 36$\times$ larger than StyleGAN2 [Kar20b] and 6$\times$ larger than StyleGAN-XL [Sau22] and XMC-GAN [Zha21n]. While our 1B parameter count is still lower than the largest recent synthesis models, such as Imagen (3.0B), DALL$\cdot$E 2 (5.5B), and Parti (20B), we have not yet observed a quality saturation regarding the model size. GigaGAN achieves a zero-shot FID of 9.09 on COCO2014 dataset, lower than the FID of DALL$\cdot$E 2, Parti-750M, and Stable Diffusion.
Furthermore, GigaGAN has three major practical advantages compared to diffusion and autoregressive models. First, it is orders of magnitude faster, generating a 512px image in 0.13 seconds ([Figure 1](#figure-01)). Second, it can synthesize ultra high-res images at 4k resolution in 3.66 seconds. Third, it is endowed with a controllable, latent vector space that lends itself to well-studied controllable image synthesis applications, such as style mixing ([Figure 6](#figure-06)), prompt interpolation ([Figure 7](#figure-07)), and prompt mixing ([Figure 8](#figure-08)).
In summary, our model is the first GAN-based method that successfully trains a billion-scale model on billions of real-world complex Internet images. This suggests that GANs are still a viable option for text-to-image synthesis and should be considered for future aggressive scaling. Please visit our [website](https://mingukkang.github.io/GigaGAN/) for additional results.
<span id="section-2"></span>
## 2 Related Works
**Text-to-image synthesis.**
Generating a realistic image given a text description, explored by early works [Zhu07, Man16], is a challenging task. A common approach is text-conditional GANs [Ree16a, Ree16, Zha17c, Xu18a, Zhu19b, Tao22] on specific domains [Wel10] and datasets with a closed-world assumption [Lin14a]. With the development of diffusion models [Den20, Dha21], autoregressive (AR) transformers [Che20b], and large-scale language encoders [Raf20a, Rad21], text-to-image synthesis has shown remarkable improvement on an open-world of arbitrary text descriptions. GLIDE [Nic22], DALL$\cdot$E 2 [Ram22], and Imagen [Sah22a] are representative diffusion models that show photorealistic outputs with the aid of a pretrained language encoder [Rad21, Raf20a]. AR models, such as DALL$\cdot$E [Ram21], Make-A-Scene [Gaf22], CogView [Din21, Din22a], and Parti [Yu22] also achieve amazing results. While these models exhibit unprecedented image synthesis ability, they require time-consuming iterative processes to achieve high-quality image sampling.
To accelerate the sampling, several methods propose to reduce the sampling steps [Men22a, Sal22, Son21a, Lu22c] or reuse pre-computed features [Li22g]. Latent Diffusion Model (LDM) [Rom22] performs the reverse processes in low-dimensional latent space instead of pixel space. However, consecutive reverse processes are still computationally expensive, limiting the usage of large-scale text-to-image models for interactive applications.
**GAN-based image synthesis.**
GANs [Goo14a] have been one of the primary families of generative models for natural image synthesis. As the sampling quality and diversity of GANs improve [Rad15, Kar18a, Kar19, Kar20b, Kar21, Kum22a, Sau21], GANs have been deployed to various computer vision and graphics applications, such as text-to-image synthesis [Ree16], image-to-image translation [Iso17, Zhu17, Hua18b, Lee18d, Par19c, Par20c], and image editing [Zhu16, Bro17, Abd19a, Pat21]. Notably, StyleGAN-family models [Kar20b, Kar21] have shown impressive ability in image synthesis tasks for single-category domains [Abd19a, Zhu20a, Wul20, Har20c, Pat21]. Other works have explored class-conditional GANs [Zha19m, Bro19, Kan21, Zha20c, Sau22] on datasets with a fixed set of object categories.
In this paper, we change the data regimes from single- or multi-categories datasets to extremely data-rich situations. We make the first expedition toward training a large-scale GAN for text-to-image generation on a vast amount of web-crawled text and image pairs, such as LAION2B-en [Sch22b] and COYO-700M [Bye22]. Existing GAN-based text-to-image synthesis models [Ree16, Zha17c, Xu18a, Zhu19b, Lia20, Zha21n, Tao22] are trained on relatively small datasets, such as CUB-200 (12k training pairs), MSCOCO (82k) and LN-OpenImages (507k). Also, those models are evaluated on associated validation datasets, which have not been validated to perform large-scale text-image synthesis like diffusion or AR models.
Concurrent with our method, StyleGAN-T [Sau23] and GALIP [Tao23a] share similar goals and make complementary insights to ours.
**Super-resolution for large-scale text-to-image models.**
Large-scale models require prohibitive computational costs for both training and inference. To reduce the memory and running time, cutting-edge text-to-image models [Nic22, Ram22, Sah22a, Yu22] have adopted cascaded generation processes where images are first generated at $64\times64$ resolution and upsampled to $256\times256$ and $1024\times1024$ sequentially. However, the super-resolution networks are primarily based on diffusion models, which require many iterations. In contrast, our low-res image generators and upsamplers are based on GANs, reducing the computational costs for both stages. Unlike traditional super-resolution techniques [Don15, Led17, Wan18h, Anw20] that aim to faithfully reproduce low-resolution inputs or handle image degradation like compression artifacts, our upsamplers for large-scale models serve a different purpose. They need to perform larger upsampling factors while potentially leveraging the input text prompt.
<span id="section-3"></span>
## 3 Method
<span id="figure-04"></span>

![Figure 4](../../papers/gigagan/figure-04.png)

**Figure 4.** Our GigaGAN high-capacity text-to-image generator. First, we extract text embeddings using a pretrained CLIP model and a learned encoder $T$. The local text descriptors are fed to the generator using cross-attention. The global text descriptor, along with a latent code $\mathbf{z}$, is fed to a style mapping network $M$ to produce style code $\mathbf{w}$. The style code modulates the main generator using our style-adaptive kernel selection, shown on the right. The generator outputs an image pyramid by converting the intermediate features into RGB images. To achieve higher capacity, we use multiple attention and convolution layers at each scale ([Table A2](#table-07)). We also use a separate upsampler model, which is not shown in this diagram.
We train a generator $G(\mathbf{z}, \mathbf{c})$ to predict an image $\mathbf{x}\in \mathbb{R}^{H\times W\times 3}$ given a latent code $\mathbf{z} \sim \mathcal{N}(0, 1) \in \mathbb{R}^{128}$ and text-conditioning signal $\mathbf{c}$. We use a discriminator $D(\mathbf{x}, \mathbf{c})$ to judge the realism of the generated image, as compared to a sample from the training database $\mathcal{D}$, which contains image-text pairs.
Although GANs [Kar18a, Kar19, Bro19] can successfully generate realistic images on single- and multi-category datasets [Kar19, Yu15a, Den09a], open-ended text-conditioned synthesis on Internet images remains challenging. We hypothesize that the current limitation stems from its reliance on convolutional layers. That is, the same convolution filters are challenged to model the general image synthesis function for all text conditioning across all locations of the image. In this light, we seek to inject more expressivity into our parameterization by dynamically selecting convolution filters based on the input conditioning and by capturing long-range dependence via the attention mechanism.
Below, we discuss our key contributions to making ConvNets more expressive ([Section 3.1](#section-3-1)), followed by our designs for the generator ([Section 3.2](#section-3-2)) and discriminator ([Section 3.3](#section-3-3)). Lastly, we introduce a new, fast GAN-based upsampler model that can improve the inference quality and speed of our method and diffusion models such as Imagen [Sah22a] and DALL$\cdot$E 2 [Ram22].
<span id="section-3-1"></span>
### 3.1 Modeling complex contextual interaction
**Baseline StyleGAN generator.**
We base our architecture off the conditional version of StyleGAN2 [Kar20b], comprised of two networks $G = \widetilde{G}\circ M$. The mapping network $\mathbf{w}=M(\mathbf{z},\mathbf{c})$ maps the inputs into a “style” vector $\mathbf{w}$, which modulates a series of upsampling convolutional layers in the synthesis network $\widetilde{G}(\mathbf{w})$ to map a learned constant tensor to an output image $\mathbf{x}$. Convolution is the main engine to generate all output pixels, with the $\mathbf{w}$ vector as the only source of information to model conditioning.
**Sample-adaptive kernel selection.**
To handle the highly diverse distribution of internet images, we aim to increase the capacity of convolution kernels. However, increasing the width of the convolution layers becomes too demanding, as the same operation is repeated across all locations.
We propose an efficient way to enhance the expressivity of convolutional kernels by creating them on-the-fly based on the text conditioning, as illustrated in [Figure 4](#figure-04) (right). In this scheme, we instantiate a bank of $N$ filters $\{\mathbf{K}_i \in \mathbb{R}^{C_{\text{in}} \times C_{\text{out}} \times K \times K} \}_{i=1}^N$, instead of one, that takes a feature $\mathbf{f} \in \mathbb{R}^{C_\text{in}}$ at each layer. The style vector $\mathbf{w} \in \mathbb{R}^{d}$ then goes through an affine layer $[W_\text{filter}, b_\text{filter}] \in \mathbb{R}^{(d + 1) \times N}$ to predict a set of weights to average across the filters, to produce an aggregated filter $\mathbf{K} \in \mathbb{R}^{C_{\text{in}} \times C_{\text{out}} \times K \times K}$.
<span id="equation-01"></span>

$$
\mathbf{K} = \sum_{i=1}^N \mathbf{K}_i \cdot \text{softmax}\big( W_\text{filter}^{\top}\mathbf{w} + b_\text{filter}\big)_i
$$
The filter is then used in the regular convolution pipeline of StyleGAN2, with the second affine layer $[W_\text{mod}, b_\text{mod}] \in \mathbb{R}^{(d + 1) \times C_\text{in}}$ for weight (de-)modulation [Kar20b].
$$
g_\text{adaconv}(\mathbf{f}, \mathbf{w}) = \big((W_{\text{mod}}^{\top}\mathbf{w} + b_\text{mod} \big) \otimes \mathbf{K}) *\mathbf{f},
$$
where $\otimes$ and $*$ represent (de-)modulation and convolution.
At a high level, the softmax-based weighting can be viewed as a differentiable filter selection process based on input conditioning. Furthermore, since the filter selection process is performed only once at each layer, the selection process is much faster than the actual convolution, decoupling compute complexity from the resolution. Our method shares a spirit with dynamic convolutions [Jia16, Wu18d, Ha17, Tan20b] in that the convolution filters dynamically change per sample, but differs in that we explicitly instantiate a larger filter bank and select weights based on a separate pathway conditional on the $\mathbf{w}$-space of StyleGAN.
**Interleaving attention with convolution.**
Since the convolutional filter operates within its receptive field, it cannot contextualize itself in relationship to distant parts of the images. One way to incorporate such long-range relationships is using attention layers $g_\text{attention}$. While recent diffusion-based models [Dha21, Ho22b, Rom22] have commonly adopted attention mechanisms, StyleGAN architectures are predominantly convolutional with the notable exceptions such as BigGAN [Bro19], GANformer [Hud21], and ViTGAN [Lee22a].
We aim to improve the performance of StyleGAN by integrating attention layers with the convolutional backbone. However, simply adding attention layers to StyleGAN often results in training collapse, possibly because the dot-product self-attention is not Lipschitz, as pointed out by Kim et al. [Kim21b]. As the Lipschitz continuity of discriminators has played a critical role in stable training [Arj17, Gul17, Mes18], we use the L2-distance instead of the dot product as the attention logits to promote Lipschitz continuity [Kim21b], similar to ViTGAN [Lee22a].
To further improve performance, we find it crucial to match the architectural details of StyleGAN, such as equalized learning rate [Kar18a] and weight initialization from a unit normal distribution. We scale down the L2 distance logits to roughly match the unit normal distribution at initialization and reduce the residual gain from the attention layers. We further improve stability by tying the key and query matrix [Lee22a], and applying weight decay.
In the synthesis network $\widetilde{G}$, the attention layers are interleaved with each convolutional block, leveraging the style vector $\mathbf{w}$ as an additional token. At each attention block, we add a separate cross-attention mechanism $g_\text{cross-attention}$ to attend to individual word embeddings [Bah15]. We use each input feature tensor as the query, and the text embeddings as the key and value of the attention mechanism.
<span id="section-3-2"></span>
### 3.2 Generator design
**Text and latent-code conditioning.**
First, we extract the text embedding from the prompt. Previous works [Ram21, Sah22a] have shown that leveraging a strong language model is essential for producing strong results. To do so, we tokenize the input prompt (after padding it to $C=77$ words, following best practices [Ram21, Sah22a]) to produce conditioning vector $\mathbf{c}\in \mathbb{R}^{C\times 768}$, and take the features from the penultimate layer [Sah22a] of a frozen CLIP feature extractor [Rad21]. To allow for additional flexibility, we apply additional attention layers $T$ on top to process the word embeddings before passing them to the MLP-based mapping network. This results in text embedding $\mathbf{t}=T(\mathcal{E}_{\text{txt}}(\mathbf{c})) \in \mathbb{R}^{C\times 768}$. Each component $\mathbf{t}_i$ of $\mathbf{t}$ captures the embedding of the $i^{\text{th}}$ word in the sentence. We refer to them as $\mathbf{t}_\text{local} = \mathbf{t}_{\{1:C\} \setminus \text{EOT}} \in \mathbb{R}^{(C - 1)\times 768}$. The EOT (“end of text") component of $\mathbf{t}$ aggregates global information, and is called $\mathbf{t}_\text{global} \in \mathbb{R}^{768}$. We process this global text descriptor, along with the latent code $\mathbf{z} \sim \mathcal{N}(0, 1)$, via an MLP mapping network to extract the style $\mathbf{w} = M(\mathbf{z},\mathbf{t}_\text{global})$.
$$
\begin{split}
    (\mathbf{t}_\text{local}&, \mathbf{t}_\text{global}) = T(\mathcal{E}_{\text{txt}}(\mathbf{c})), \\
    &\mathbf{w} = M(\mathbf{z},\mathbf{t}_\text{global}).
\end{split}
$$
Different from the original StyleGAN, we use both the text-based style code $\mathbf{w}$ to modulate the synthesis network $\widetilde{G}$ and the word embeddings $\mathbf{t}_\text{local}$ as features for cross-attention.
$$
\mathbf{x} = \widetilde{G}(\mathbf{w}, \mathbf{t}_\text{local}).
$$
Similar to earlier works [Man16, Sah22a, Ram22], the text-image alignment visually improves with cross-attention.
**Synthesis network.**
Our synthesis network consists of a series of upsampling convolutional layers, with each layer enhanced with the adaptive kernel selection ([Equation 1](#equation-01)) and followed by our attention layers.
$$
\hspace{-1ex}\mathbf{f}_{\ell+1}=g^\ell_\text{xa}(g^\ell_\text{attn}(g^\ell_\text{adaconv}(\mathbf{f}_\ell, \mathbf{w}), \mathbf{w}), \mathbf{t}_\text{local}),
$$
where $g^\ell_\text{xa}$, $g^\ell_\text{attn}$, and $g^\ell_\text{adaconv}$ denote the $l$-th layer of cross-attention, self-attention, and weight (de-)modulation layers. We find it beneficial to increase the depth of the network by adding more blocks at each layer. In addition, our generator outputs a multi-scale image pyramid with $L=5$ levels, instead of a single image at the highest resolution, similar to MSG-GAN [Kar20c] and AnycostGAN [Lin21b]. We refer to the pyramid as $\{\mathbf{x}_i\}_{i=0}^{L-1} = \{\mathbf{x}_0, \mathbf{x}_1, ..., \mathbf{x}_4\}$, with spatial resolutions $\{S_i\}_{i=0}^{L-1} = \{64, 32, 16, 8, 4\}$, respectively. The base level $\mathbf{x}_0$ is the output image $\mathbf{x}$. Each image of the pyramid is independently used to compute the GAN loss, as discussed in [Section 3.3](#section-3-3). We follow the findings of StyleGAN-XL [Sau22] and turn off the style mixing and path length regularization [Kar20b]. We include more training details in [Section 6.1](#section-6-1).
<span id="section-3-3"></span>
### 3.3 Discriminator design
As shown in [Figure 5](#figure-05), our discriminator consists of separate branches for processing text with the function $\mathbf{t}_D$ and images with function $\phi$. The prediction of real vs. fake is made by comparing the features from the two branches using function $\psi$. We introduce a new way of making predictions on multiple scales. Finally, we use additional CLIP and Vision-Aided GAN losses [Kum22a] to improve stability.
**Text conditioning.**
First, to incorporate conditioning into discriminators, we extract text descriptor $\mathbf{t}_\text{D}$ from text $\mathbf{c}$. Similar to the generator, we apply a pretrained text encoder, such as CLIP [Rad21], followed by a few learnable attention layers. In this case, we only use the global descriptor.
<span id="figure-05"></span>

![Figure 5](../../papers/gigagan/figure-05.png)

**Figure 5.** Our discriminator consists of two branches for processing the image and the text conditioning $t_D$. The text branch processes the text similar to the generator ([Figure 4](#figure-04)). The image branch receives an image pyramid and makes independent predictions for each image scale. Moreover, the predictions are made at all subsequent scales of the downsampling layers, making it a multi-scale input, multi-scale output (MS-I/O) discriminator.
**Multiscale image processing.**
We observe that the early, low-resolution layers of the generator become inactive, using small dynamic ranges irrespective of the provided prompts. StyleGAN2 [Kar20b] also observes this phenomenon, concluding that the network relies on the high-resolution layers, as the model size increases. As recovering performance in low frequencies, which contains complex structure information, is crucial, we redesign the model architecture to provide training signals across multiple scales.
Recall the generator produces a pyramid $\{\mathbf{x}_i\}_{i=0}^{L-1}$, with the full image $\mathbf{x}_0$ at the pyramid base. MSG-GAN [Kar20c] improves performance by making a prediction on the entire pyramid at once, enforcing consistency across scales. However, in our large-scale setting, this harms stability, as this limits the generator from making adjustments to its initial low-res output.
Instead, we process each level of the pyramid *independently*. As shown in [Figure 5](#figure-05), each level $\mathbf{x}_i$ makes a real/fake prediction at multiple scales $i < j \leq L$. For example, the full $\mathbf{x}_0$ makes predictions at $L=5$ scales, the next level $\mathbf{x}_1$ makes predictions at 4 scales, and so on. In total, our discriminator produces $\frac{L(L+1)}{2}$ predictions, supervising multi-scale generations at multiple scales.

To extract features at different scales, we define feature extractor $\phi_{i\rightarrow j}: \mathbb{R}^{X_i\times X_i\times 3} \rightarrow \mathbb{R}^{X_j^D\times X_j^D\times C_j}$. Practically, each sub-network $\phi_{i\rightarrow j}$ is a subset of full $\phi\triangleq \phi_{0\rightarrow L}$, with $i>0$ indicating late entry and $j<L$ indicating early exit. Each layer in $\phi$ is composed of self-attention, followed by convolution with stride 2. The final layer flattens the spatial extent into a $1\times 1$ tensor. This produces output resolutions at $\{X_j^D\} = \{32, 16, 8, 4, 1\}$. This allows us to inject lower-resolution images on the pyramid into intermediate layers [Kar18a]. As we use a shared feature extractor across different levels and most of the added predictions are made at low resolutions, the increased computation overhead is manageable.
**Multi-scale input, multi-scale output adversarial loss.** In total, our training objective consists of discriminator losses, along with our proposed matching loss, to encourage the discriminator to take into account the conditioning:

<span id="equation-02"></span>

$$
\mathcal{V}_\text{MS-I/O}(G, D) = \sum_{i=0}^{L-1} \sum_{j=i+1}^{L} \mathcal{V}_\text{GAN}(G_i, D_{ij}) + \mathcal{V}_\text{match}(G_i, D_{ij}),
$$

where $\mathcal{V}_\text{GAN}$ is the standard, non-saturating GAN loss [Goo14a].

To compute the discriminator output, we train predictor $\psi$, which uses text feature $\mathbf{t}_\text{D}$ to modulate image features $\phi(\mathbf{x})$:

<span id="equation-03"></span>

$$
D_{ij}(\mathbf{x}, \mathbf{c}) = \psi_j(\phi_{i\rightarrow j}(\mathbf{x}_i), \mathbf{t}_D) + \text{Conv}_{1\times 1}(\phi_{i\rightarrow j}(\mathbf{x}_i)),
$$

where $\psi_j$ is implemented as a 4-layer $1\times 1$ modulated convolution, and $\text{Conv}_{1\times 1}$ is added as a skip connection to explicitly maintain an unconditional prediction branch [Miy18].

**Matching-aware loss.** The previous GAN terms measure how closely the image $\mathbf{x}$ matches the conditioning $\mathbf{c}$, as well as how realistic $\mathbf{x}$ looks, irrespective of conditioning. However, during early training, when artifacts are obvious, the discriminator heavily relies on making a decision independent of conditioning and hesitates to account for the conditioning.

To enforce the discriminator to incorporate conditioning, we match $\mathbf{x}$ with a random, independently sampled condition $\hat{\mathbf{c}}$, and present them as a fake pair:

<span id="equation-04"></span>

$$
\mathcal{V}_{\text{match}} = \mathbb{E}_{\mathbf{x},\mathbf{c},\hat{\mathbf{c}}} \Big[ \log(1 + \exp(D(\mathbf{x}, \hat{\mathbf{c}}))) + \log(1 + \exp(D(G(\mathbf{c}), \hat{\mathbf{c}}))) \Big],
$$

where $(\mathbf{x}, \mathbf{c})$ and $\hat{\mathbf{c}}$ are separately sampled from $p_{\text{data}}$. This loss has previously been explored in text-to-image GAN works [Ree16, Zha17c], except we find that enforcing the matching-aware loss on generated images from $G$, as well as real images $\mathbf{x}$, leads to clear gains in performance ([Table 1](#table-01)).

**CLIP contrastive loss.** We further leverage off-the-shelf pretrained models as a loss function [Sau21, Kum22a, Sun18]. In particular, we enforce the generator to produce outputs that are identifiable by the pre-trained CLIP image and text encoders [Rad21], $\mathcal{E}_\text{img}$ and $\mathcal{E}_\text{txt}$, in the contrastive cross-entropy loss that was used to train them originally.

<span id="equation-05"></span>

$$
\mathcal{L}_\text{CLIP} = \mathbb{E}_{\{\mathbf{c}_n\}} \Bigg[ -\log \frac{\exp(\mathcal{E}_\text{img}(G(\mathbf{c}_0))^\top \mathcal{E}_\text{txt}(\mathbf{c}_0))}{\sum_n \exp(\mathcal{E}_\text{img}(G(\mathbf{c}_0))^\top \mathcal{E}_\text{txt}(\mathbf{c}_n))} \Bigg].
$$

where $\{\mathbf{c}_n\} = \{\mathbf{c}_0, \dots\}$ are sampled captions from the training data.

**Vision-aided adversarial loss.** Lastly, we build an additional discriminator that uses the CLIP model as a backbone, known as Vision-Aided GAN [Kum22a]. We freeze the CLIP image encoder, extract features from the intermediate layers, and process them through a simple network with $3\times 3$ convolutional layers to make real/fake predictions. We also incorporate conditioning through modulation, as in [Equation 3](#equation-03). To stabilize training, we also add a fixed random projection layer, as proposed by Projected GAN [Sau21]. We refer to this as $\mathcal{L}_\text{Vision}(G)$ (omitting the learnable discriminator parameters for clarity).

Our final objective is $\mathcal{V}(G, D) = \mathcal{V}_\text{MS-I/O}(G,D) + \mathcal{L}_\text{CLIP}(G) + \mathcal{L}_\text{Vision}(G)$, with weighting between the terms specified in [Table A2](#table-07).
<span id="section-3-4"></span>
### 3.4 GAN-based upsampler
Furthermore, GigaGAN framework can be easily extended to train a text-conditioned superresolution model, capable of upsampling the outputs of the base GigaGAN generator to obtain high-resolution images at 512px or 2k resolution. By training our pipeline in two separate stages, we can afford a higher capacity 64px base model within the same computational resources.
In the upsampler, the synthesis network is rearranged to an asymmetric U-Net architecture, which processes the 64px input through 3 downsampling residual blocks, followed by 6 upsampling residual blocks with attention layers to produce the 512px image. There exist skip connections at the same resolution, similar to CoModGAN [Zha21o]. The model is trained with the same losses as the base model, as well as the LPIPS Perceptual Loss [Zha18d] with respect to the ground truth high-resolution image. Vision-aided GAN is not used for the upsampler. During training and inference time, we apply moderate Gaussian noise augmentation to reduce the gap between real and GAN-generated images. Please refer to [Section 6.3](#section-6-3) for more details.
Our GigaGAN framework becomes particularly effective for the superresolution task compared to the diffusion-based models, which cannot afford as many sampling steps as the base model at high resolution. The LPIPS regression loss also provides a stable learning signal. We believe that our GAN upsampler can serve as a drop-in replacement for the superresolution stage of other generative models.
<span id="section-4"></span>
## 4 Experiments
<span id="figure-06"></span>

![Figure 6](../../papers/gigagan/figure-06.png)

**Figure 6.** Style mixing. Our GAN-based architecture retains a disentangled latent space, enabling us to blend the coarse style of one sample with the fine style of another. All outputs are generated with the prompt “A Toy sport sedan, CG art." The corresponding latent codes are spliced together to produce a style-swapping grid.
<span id="figure-07"></span>

![Figure 7](../../papers/gigagan/figure-07.png)

**Figure 7.** Prompt interpolation. GigaGAN enables smooth interpolation between prompts, as shown in the interpolation grid. The four corners are generated from the same latent $\mathbf{z}$ but with different text prompts. The corresponding text embeddings $\mathbf{t}$ and style vectors $\mathbf{w}$ are interpolated to create a smooth transition. The same $\mathbf{z}$ results in similar layouts. See [Figure 8](#figure-08) for more precise control.
<span id="figure-08"></span>

![Figure 8](../../papers/gigagan/figure-08.png)

**Figure 8.** Prompt mixing. GigaGAN retains a disentangled latent space, enabling us to combine the coarse style of one sample with the fine style of another. Moreover, GigaGAN can directly control the style with text prompts. Here we generate four outputs using the prompts “a X on tabletop", shown in the “no mixing” column. Then we re-compute the text embeddings $\mathbf{t}$ and the style codes $\mathbf{w}$ using the new prompts “a X with the texture of Y on tabletop", such as “a cube with the texture of crochet on tabletop", and apply them to the second half layers of the generator, achieving layout-preserving fine style control. Cross-attention mechanism automatically localizes the style to the object of interest.
Systematic, controlled evaluation of large-scale text-to-image synthesis tasks is difficult, as most existing models are not publicly available. Training a new model from scratch would be prohibitively costly, even if the training code were available. Still, we compare our model to recent text-to-image models, such as Imagen [Sah22a], Latent Diffusion Models (LDM) [Rom22], Stable Diffusion [Sta22], and Parti [Yu22], based on the available information, while acknowledging considerable differences in the training dataset, number of iterations, batch size, and model size. In addition to text-to-image results, we evaluate our model on ImageNet class-conditional generation in [Section 7](#section-7), for an apples-to-apples comparison with other methods at a more controlled setting.
For quantitative evaluation, we mainly use the Fréchet Inception Distance(FID) [Heu17] for measuring the realism of the output distribution and the CLIP score for evaluating the image-text alignment.
We conduct five different experiments. First, we show the effectiveness of our method by gradually incorporating each technical component one by one ([Section 4.2](#section-4-2)). Second, our text-to-image synthesis results demonstrate that GigaGAN exhibits comparable FID with Stable Diffusion (SD-v1.5) [Rom22] while generating results hundreds of times faster than diffusion or autoregressive models ([Section 4.3](#section-4-3)). Third, we compare GigaGAN with a distillation-based diffusion model [Men22a] and show that GigaGAN can synthesize higher-quality images faster than the distillation-based diffusion model. Fourth, we verify the advantage of GigaGAN’s upsampler over other upsamplers in both conditional and unconditional super-resolution tasks. Lastly, we show our large-scale GANs still enjoy the continuous and disentangled latent space manipulation of GANs, enabling new image editing modes ([Section 4.6](#section-4-6)).
<span id="section-4-1"></span>
### 4.1 Training and evaluation details
We implement GigaGAN based on the StudioGAN PyTorch library [Kan22], following the standard FID evaluation protocol with the anti-aliasing bicubic resize function [Par22c], unless otherwise noted. For text-to-image synthesis, we train our models on the union of LAION2B-en [Sch22b] and COYO-700M [Bye22] datasets, with the exception of the 128-to-1024 upsampler model trained on Adobe’s internal Stock images. The image-text pairs are preprocessed based on CLIP score [Hes21], image resolution, and aesthetic score [Sch00], similar to prior work [Sta22]. We use CLIP ViT-L/14 [Rad21] for the pre-trained text encoder and OpenCLIP ViT-G/14 [Ilh21] for CLIP score calculation [Hes21] except for [Table 1](#table-01). All our models are trained and evaluated on A100 GPUs. We include more training and evaluation details in [Section 6](#section-6).
<span id="section-4-2"></span>
### 4.2 Effectiveness of proposed components

<span id="table-01"></span>

![Table 1](../../papers/gigagan/table-01.png)

**Table 1.** Ablation study on 64px text-to-image synthesis. To evaluate the effectiveness of our components, we start with a modified version of StyleGAN for text conditioning. While increasing the network width does not show satisfactory improvement, each addition of our contributions keeps improving metrics. Finally, we increase the network width and scale up training to reach our final model. All ablated models are trained for 100k steps at a batch size of 256 except for the Scale-up row (1350k iterations with a larger batch size). CLIP Score is computed using CLIP ViT-B/32 [Rad21].
First, we show the effectiveness of our formulation via ablation study in [Table 1](#table-01). We set up a baseline by adding text-conditioning to StyleGAN2 and tuning the configuration based on the findings of StyleGAN-XL. We first directly increase the model size of this baseline, but we find that this does not improve the FID and CLIP scores. Then, we add our components one by one and observe that they consistently improve performance. In particular, our model is more scalable, as the higher-capacity version of the final formulation achieves better performance.
<span id="section-4-3"></span>
### 4.3 Text-to-Image synthesis
We proceed to train a larger model by increasing the capacity of the base generator and upsampler to 652.5M and 359.1M, respectively. This results in an unprecedented size of GAN model, with a total parameter count of 1.0B. [Table 2](#table-02) compares the performance of our end-to-end pipeline to various text-to-image generative models [Ram21, Nic22, Rom22, Ram22, Sah22a, Bal22, Yu22, Zho22d, Sta22, Cha23e]. Note that there exist differences in the training dataset, the pretrained text encoders, and even image resolutions. For example, GigaGAN initially synthesizes 512px images, which are resized to 256px before evaluation.

<span id="table-02"></span>

![Table 2](../../papers/gigagan/table-02.png)

**Table 2.** Comparison on 256px text-to-image generation using COCO zero-shot FID with 30k samples. LDM [Rom22], Stable Diffusion [Sta22], and Parti [Yu22] provide models of comparable size to ours. [Base] and [Upsampler] indicate model size of each component. [Steps] indicates the number of forward passes of the model at inference. [MAC] measures the multiply-accumulate op count.
[Table 2](#table-02) shows that GigaGAN exhibits a lower FID than DALL$\cdot$E 2 [Ram22], Stable Diffusion [Sta22], and Parti-750M [Yu22]. While our model can be optimized to better match the feature distribution of real images than existing models, the quality of the generated images is not necessarily better (see [Section 8](#section-8) for more samples). We acknowledge that this may represent a corner case of zero-shot FID on COCO2014 dataset and suggest that further research on a better evaluation metric is necessary to improve text-to-image models. Nonetheless, we emphasize that GigaGAN is the first GAN model capable of synthesizing promising images from arbitrary text prompts and exhibits competitive zero-shot FID with other text-to-image models.
<span id="section-4-4"></span>
### 4.4 Comparison with distilled diffusion models
While GigaGAN is at least 20 times faster than the above diffusion models, there have been efforts to improve the inference speed of diffusion models. We compare GigaGAN with progressively distilled Stable Diffusion (SD-distilled) [Men22a].
<span id="table-03"></span>

![Table 3](../../papers/gigagan/table-03.png)

**Table 3.** Comparison to distilled diffusion models shows that GigaGAN achieves better FID and CLIP scores compared to the progressively distilled diffusion models [Men22a] for fast inference. As GigaGAN generates outputs in a single feedforward pass, the inference speed is still faster. The evaluation setup is different from [Table 2](#table-02) to match SD-distilled's protocol [Men22a].
[Table 3](#table-03) demonstrates that GigaGAN remains faster than the distilled Stable Diffusion while showing better FID and CLIP scores of 21.1 and 0.32, respectively. We follow the evaluation protocol of SD-distilled [Men22a] and report FID and CLIP scores on COCO2017 dataset [Lin14a], where images are resized to 512px.
<span id="section-4-5"></span>
### 4.5 Super-resolution for large-scale image synthesis
We separately evaluate the performance of the GigaGAN upsampler. Our evaluation consists of two parts. First, we compare GigaGAN with several commonly-used upsamplers. For the text-conditioned upsampling task, we combine the Stable Diffusion [Sta22] 4x Upscaler and 2x Latent Upscaler to establish an 8x upscaling model (SD Upscaler). We also use the unconditional Real-ESRGAN [Wan21i] as another baseline.
<span id="table-04"></span>

![Table 4](../../papers/gigagan/table-04.png)

**Table 4.** Text-conditioned 128→1024 super-resolution on random 10K LAION samples, compared against unconditional Real-ESRGAN [Wan21i] and Stable Diffusion Upscaler [Sta22]. GigaGAN enjoys the fast speed of a GAN-based model while achieving better FID, patch-FID [Cha22c], CLIP score, and LPIPS [Zha18d].
[Table 4](#table-04) measures the performance of the upsampler on random 10K images from the LAION dataset and shows that our GigaGAN upsampler significantly outperforms the other upsamplers in realism scores (FID and patch-FID [Cha22c]), text alignment (CLIP score) and closeness to the ground truth (LPIPS [Zha18d]). In addition, for more controlled comparison, we train our model on the ImageNet *unconditional* superresolution task and compare performance with the diffusion-based models, including SR3 [Sah22b] and LDM [Rom22].
<span id="table-05"></span>

![Table 5](../../papers/gigagan/table-05.png)

**Table 5.** Unconditional 64→256 super-resolution on ImageNet. We compare to a simple U-Net trained with a pixel regression loss (U-Net regression), and diffusion-based methods (SR3 [Sah22b] and LDM [Rom22]). Our method achieves higher realism scores represented by the Inception Score (IS) and FID.
As shown in [Table 5](#table-05), GigaGAN achieves the best IS and FID scores with a single feedforward pass.
<span id="section-4-6"></span>
### 4.6 Controllable image synthesis
StyleGANs are known to possess a linear latent space useful for image manipulation, called the $\mathcal{W}$-space. Likewise, we perform coarse and fine-grained style swapping using style vectors $\mathbf{w}$. Similar to the $\mathcal{W}$-space of StyleGAN, [Figure 6](#figure-06) illustrates that GigaGAN maintains a disentangled $\mathcal{W}$-space, suggesting existing latent manipulation techniques of StyleGAN can transfer to GigaGAN. Furthermore, our model possesses another latent space of text embedding $\mathbf{t} = [\mathbf{t}_\text{local}, \mathbf{t}_\text{global}]$ prior to $\mathcal{W}$, and we explore its potential for image synthesis. In [Figure 8](#figure-08), we show that the disentangled style manipulation can be controlled via text inputs. In detail, we can compute the text embedding $\mathbf{t}$ and style code $\mathbf{w}$ using different prompts and apply them to different layers of the generator. This way, we gain not only the coarse and fine style disentanglement but also an intuitive prompt-based maneuver in the style space.
<span id="section-5"></span>
## 5 Discussion and Limitations
<span id="figure-09"></span>

![Figure 9](../../papers/gigagan/figure-09.png)

**Figure 9.** Failure cases. Our outputs with the same prompts as DALL$\cdot$E 2. Each column conditions on “a teddy bear on a skateboard in Times Square", “a Vibrant portrait painting of Salvador Dali with a robotic half face", and “A close up of a handpalm with leaves growing from it". Compared to production-grade models such as DALL$\cdot$E 2, our model exhibits limitations in realism and compositionality. See Appendix 8 for uncurated comparisons.
Our experiments provide a conclusive answer about the scalability of GANs: our new architecture can scale up to model sizes that enable text-to-image synthesis. However, the visual quality of our results is not yet comparable to production-grade models like DALL$\cdot$E 2. [Figure 9](#figure-09) shows several instances where our method fails to produce high-quality results when compared to DALL$\cdot$E 2, in terms of photorealism and text-to-image alignment for the same input prompts used in their paper.
Nevertheless, we have tested capacities well beyond what is possible with a naïve approach and achieved competitive results with autoregressive and diffusion models trained with similar resources while being orders of magnitude faster and enabling latent interpolation and stylization. Our GigaGAN architecture opens up a whole new design space for large-scale generative models and brings back key editing capabilities that became challenging with the transition to autoregressive and diffusion models. We expect our performance to improve with larger models, as seen in [Table 1](#table-01).
## Acknowledgments

We thank Simon Niklaus, Alexandru Chiculita, and Markus Woodson for building the distributed training pipeline. We thank Nupur Kumari, Gaurav Parmar, Bill Peebles, Phillip Isola, Alyosha Efros, and Joonghyuk Shin for their helpful comments. We also want to thank Chenlin Meng, Chitwan Saharia, and Jiahui Yu for answering many questions about their fantastic work. We thank Kevin Duarte for discussions regarding upsampling beyond 4K. Part of this work was done while Minguk Kang was an intern at Adobe Research. Minguk Kang and Jaesik Park were supported by IITP grant funded by the government of South Korea (MSIT) (POSTECH GSAI: 2019-0-01906 and Image restoration: 2021-0-00537).
We first provide training and evaluation details in [Section 6](#section-6). Then, we share results on ImageNet, with visual comparison to existing methods in [Section 7](#section-7). Lastly in [Section 8](#section-8), we show more visuals on our text-to-image synthesis results and compare them with LDM [Rom22], Stable Diffusion [Sta22], and DALL$\cdot$E 2 [Ram22].
<span id="section-6"></span>
## 6 Training and evaluation details
<span id="section-6-1"></span>
### 6.1 Text-to-image synthesis

<span id="table-07"></span>

![Table A2](../../papers/gigagan/table-07.png)

**Table A2.** Hyperparameters for GigaGAN training. We denote Projection Discriminator [Miy18] as PD, R1 regularization [Mes18] as R1, Learned Perceptual Image Patch Similarity [Zha18d] as LPIPS, Adam with decoupled weight decay [Los19] as AdamW, and the pretrained VIT-B/32 visual encoder [Rad21] as CLIP-ViT-B/32-V.
We train GigaGAN on a combined dataset of LAION2B-en [Sch22b] and COYO-700M [Bye22] in PyTorch framework [Pas19a]. For training, we apply center cropping, which results in a square image whose length is the same as the shorter side of the original image. Then, we resize the image to the resolution $64\times64$ using PIL.LANCZOS [Tur90] resizer, which supports anti-aliasing [Par22c]. We filter the training image–text pairs based on image resolution ($\geq512$), CLIP score ($>0.3$) [Hes21], aesthetics score ($>5.0$) [Sch00], and remove watermarked images. We train our GigaGAN based on the configurations denoted in the fourth and fifth columns of [Table A2](#table-07).
For evaluation, we use 40,504 and 30,000 real and generated images from COCO2014 [Lin14a] validation dataset as described in Imagen [Sah22a]. We apply the center cropping and resize the real and generated images to $299\times299$ resolution using PIL.BICUBIC, suggested by clean-fid [Par22c]. We use the clean-fid library [Par22c] for FID calculation.
<span id="section-6-2"></span>
### 6.2 Conditional image synthesis on ImageNet
We follow the training and evaluation protocol proposed by Kang *et al*.. [Kan22] to make a fair comparison against other cutting-edge generative models. We use the same cropping strategy to process images for training and evaluation as in our text-to-image experiments. Then, we resize the image to the target resolution ($64\times64$ for the base generator or $256\times256$ for the super-resolution stack) using PIL.LANCZOS [Tur90] resizer, which supports anti-aliasing [Par22c]. Using the pre-processed training images, we train GigaGAN based on the configurations denoted in the second and third columns of [Table A2](#table-07).
For evaluation, we upsample the real and generated images to $299\times299$ resolution using the PIL.BICUBIC resizer. To compute FID, we generate 50k images without truncation tricks [Kar19, Bro19] and compare those images with the entire training dataset. We use the pre-calculated features of real images provided by StudioGAN [Kan22] and 50k generated images for Precision & Recall [Kyn19] calculation.
<span id="section-6-3"></span>
### 6.3 Super-resolution results
For model training, we preprocess ImageNet in the same way as in [Section 6.2](#section-6-2) and use the configuration in the last column of [Table A2](#table-07). To compare our model with SR3 [Sah22b] and LDM fairly, we follow the evaluation procedure described in SR3 and LDM papers.
<span id="section-7"></span>
## 7 ImageNet experiments
<span id="section-7-1"></span>
### 7.1 Qualitative results
We train a class-conditional GAN on the ImageNet dataset [Den09a], for which apples-to-apples comparison is possible using the same dataset and evaluation pipeline. Our GAN achieves comparable generation quality to the cutting-edge generative models without a pretrained ImageNet classifier, which acts favorably toward automated metrics [Kyn22]. We apply L2 self-attention, style-adaptive convolution kernel, and matching-aware loss to our model and use a wider synthesis network to train the base 64px model with a batch size of 1024. Additionally, we train a separate 256px class-conditional upsampler model and combine them with an end-to-end finetuning stage.
<span id="table-06"></span>

![Table A1](../../papers/gigagan/table-06.png)

**Table A1.** Class-conditional synthesis on ImageNet 256px. Our method performs competitively against large diffusion and transformer models. Shaded methods leverage a pretrained ImageNet classifier at training or inference time, which could act favorably toward the automated metrics [Kyn22].
[Table A1](#table-06) shows that our method generates high-fidelity images.
<span id="section-7-2"></span>
### 7.2 Quantitative results
We provide visual results from ADM-G-U, LDM, StyleGAN-XL [Sau22], and GigaGAN in [Figure A1](#figure-10) and [Figure A2](#figure-11). Although StyleGAN-XL has the lowest FID, its visual quality appears worse than ADM and GigaGAN. StyleGAN-XL struggles to synthesize the overall image structure, leading to less realistic images. In contrast, GigaGAN appears to synthesize the overall structure better than StyleGAN-XL and faithfully captures fine-grained details, such as the wing patterns of a monarch and the white fur of an arctic fox. Compared to GigaGAN, ADM-G-U synthesizes the image structure more rationally but lacks in reflecting the aforementioned fine-grained details.
<span id="figure-10"></span>

![Figure A1](../../papers/gigagan/figure-10.png)

**Figure A1.** Uncurated images (above: Tench and below: Monarch) from ADM-G-U [Dha21], LDM-4-G [Rom22], GigaGAN (ours), and StyleGAN-XL [Sau22]. FID values of each generative model are 4.01, 3.60, 3.45, and 2.32, respectively.
<span id="figure-11"></span>

![Figure A2](../../papers/gigagan/figure-11.png)

**Figure A2.** Uncurated images (above: Lorikeet and below: Arctic fox) from ADM-G-U [Dha21], LDM-4-G [Rom22], GigaGAN (ours), and StyleGAN-XL [Sau22]. FID values of each generative model are 4.01, 3.60, 3.45, and 2.32, respectively.
<span id="section-8"></span>
## 8 Text-to-image synthesis results
<span id="section-8-1"></span>
### 8.1 Truncation trick at inference
Similar to the classifier guidance [Dha21] and classifier-free guidance [Ho22d] used in diffusion models such as LDM, our GAN model can leverage the truncation trick [Bro19, Kar19] at inference time.
$$
\mathbf{w_{\text{trunc}}} = \text{lerp}(\mathbf{w_\text{mean}}, \mathbf{w}, \psi),
$$
where $\mathbf{w_\text{mean}}$ is the mean of $\mathbf{w}$ of the entire dataset, which can be precomputed. In essence, the truncation trick lets us trade diversity for fidelity by interpolating the latent vector to the mean of the distribution and thereby making the outputs more typical. When $\psi = 1.0$, $\mathbf{w_\text{mean}}$ is not used, and there is no truncation. When $\psi = 0.0$, $\mathbf{w}$ collapses to the mean, losing diversity.
While it is straightforward to apply the truncation trick for the unconditional case, it is less clear how to achieve this for text-conditional image generation. We find that interpolating the latent vector toward both the mean of the entire distribution as well as the mean of $\mathbf{w}$ conditioned on the text prompt produces desirable results.
$$
\mathbf{w_{\text{trunc}}} = \text{lerp}(\mathbf{w_{\text{mean}, \mathbf{c}}}, \text{lerp}( \mathbf{w_\text{mean}}, \mathbf{w}, \psi ), \psi ),
$$
where $\mathbf{w}_{\text{mean}, \mathbf{c}}$ can be computed at inference time by sampling $\mathbf{w} = M(\mathbf{z}, \mathbf{c})$ 16 times with the same $\mathbf{c}$, and taking the average. This operation’s overhead is negligible, as the mapping network $M$ is computationally light compared to the synthesis network. At $\psi = 1.0$, $\mathbf{w_{\text{trunc}}}$ becomes $\mathbf{w_{\text{trunc}}} = \mathbf{w}$, meaning no truncation. [Figure A4](#figure-13) demonstrates the effect of our text-conditioned truncation trick.
Quantitatively, the effect of truncation is similar to the guidance technique of diffusion models. As shown in [Figure A3](#figure-12), the CLIP score increases with more truncation, where the FID increases due to reduced diversity.
<span id="figure-12"></span>

![Figure A3](../../papers/gigagan/figure-12.png)

**Figure A3.** We investigate how our FID and CLIP score changes over different truncation values [1.0, 0.9, 0.8, 0.7, 0.6, 0.5], by visualizing them along with the FID-CLIP score curve of two publicly available large scale diffusion models: LDM and Stable Diffusion. It is seen that the CLIP score increases with more truncation, at the cost of reduced diversity indicated by higher FID. The guidance values of the diffusion models are [1.0, 1.25, 1.5, 1.75, 2, 4, 6, 8, 10].
<span id="figure-13"></span>

![Figure A4](../../papers/gigagan/figure-13.png)

**Figure A4.** The visual effect of our truncation trick. We demonstrate the effect of truncation by decreasing the truncation value $\psi$ from 1.0. We show six example outputs with the text prompt “digital painting of a confident and severe looking northern war goddess, extremely long blond braided hair, beautiful blue eyes and red lips." and “Magritte painting of a clock on a beach.". At 1.0 (no truncation), the diversity is high, but the alignment is not satisfactory. As the truncation increases, text-image alignment improves, at the cost of diversity. We find that a truncation value between 0.8 and 0.7 produces the best result.
<span id="section-8-2"></span>
### 8.2 Comparison to diffusion models
Finally, we show randomly sampled results of our model and compare them with publicly available diffusion models, LDM [Rom22], Stable Diffusion [Sta22], and DALL$\cdot$E 2 [Ram22].
<span id="figure-14"></span>

![Figure A5](../../papers/gigagan/figure-14.png)

**Figure A5.** Style mixing. GigaGAN maintains a disentangled latent space, allowing us to blend the coarse style of one sample with the fine style of another. The corresponding latent codes are spliced together to produce a style-swapping grid. The outputs are generated from the same prompt but with different latent codes.
<span id="figure-15"></span>

![Figure A6](../../papers/gigagan/figure-15.png)

**Figure A6.** Prompt interpolation. GigaGAN enables smooth interpolation between prompts, as shown in the interpolation grid. The four corners are generated from the same latent but with different text prompts. The corresponding text embeddings and style vectors are interpolated to create a smooth transition. The same results in similar layouts.
<span id="figure-16"></span>

![Figure A7](../../papers/gigagan/figure-16.png)

**Figure A7.** Prompt mixing. GigaGAN can directly control the style with text prompts. Here we generate five outputs using the prompts “Photography of X", shown in the “Photography” column. Then we re-compute the text embeddings $\mathbf{t}$ and the style codes $\mathbf{w}$ using the new prompts “Y of X", such as “Van Gogh painting of the black hole in the space", and apply them to the second half layers of the generator, achieving layout-preserving style control. Cross-attention mechanism automatically localizes the style to the object of interest. We use the following prompts in order from the row above. (1) the black hole in the space. (2) a teacup on the desk. (3) a table top with a vase of flowers on it. (4) a birthday cake. (5) a beautiful flower. We discover that GigaGAN’s prompt-based style transfer is only possible for images of a single and simple object.
<span id="figure-17"></span>

![Figure A8](../../papers/gigagan/figure-17.png)

**Figure A8.** Diversity of synthesized images using GigaGAN. GigaGAN can synthesize diverse images for a given prompt. We use the following prompts in order from the row above. (1) Majestic castle and fireworks, art work, oil painting, 2k. (2) Oil-painting depicting a sunset over the sea with waves. (3) A burning candle with tho wicks, detailed photo, studio lighting. (4) Portrait of Isaac Newton, long hair. (5) An abstract representation of the speed of light.
<span id="figure-18"></span>

![Figure A9](../../papers/gigagan/figure-18.png)

**Figure A9.** Random outputs of our model, Latent Diffusion Model [Rom22], Stable Diffusion [Sta22], and DALL$\cdot$E 2 [Ram22], using prompt “A loft bed with a dresser underneath it". We show two versions of our model, one without truncation and the other with truncation. Our model enjoys faster speed than the diffusion models. Still, we observe our model falls behind in structural coherency, such as the number of legs of the bed frames. For LDM and Stable Diffusion, we use 250 and 50 sampling steps with DDIM / PLMS [Liu22h], respectively. For DALL$\cdot$E 2, we generate images using the official DALL$\cdot$E service [Dal22].
<span id="figure-19"></span>

![Figure A10](../../papers/gigagan/figure-19.png)

**Figure A10.** Random outputs of our model, Latent Diffusion Model [Rom22], Stable Diffusion [Sta22], and DALL$\cdot$E 2 [Ram22], using prompt “A green vase filed with red roses sitting on top of table". We show two versions of our model, one without truncation and the other with truncation. Our model enjoys faster speed than the diffusion models in both cases. Still, we observe our model falls behind in structural coherency like the symmetry of the vases. For LDM and Stable Diffusion, we use 250 and 50 sampling steps with DDIM / PLMS [Liu22h], respectively. For DALL$\cdot$E 2, we generate images using the official DALL$\cdot$E service [Dal22].
<span id="figure-20"></span>

![Figure A11](../../papers/gigagan/figure-20.png)

**Figure A11.** Random outputs of our model, Latent Diffusion Model [Rom22], Stable Diffusion [Sta22], and DALL$\cdot$E 2 [Ram22], using prompt “A zebra in the grass who is cleaning himself". We show two versions of our model, one without truncation and the other with truncation. Our model enjoys faster speed than the diffusion models in both cases. Still, we observe our model falls behind in details, such as the precise stripe pattern of the positioning of eyes. For LDM and Stable Diffusion, we use 250 and 50 sampling steps with DDIM / PLMS [Liu22h], respectively. For DALL$\cdot$E 2, we generate images using the official DALL$\cdot$E service [Dal22].
<span id="figure-21"></span>

![Figure A12](../../papers/gigagan/figure-21.png)

**Figure A12.** Random outputs of our model, Latent Diffusion Model [Rom22], Stable Diffusion [Sta22], and DALL$\cdot$E 2 [Ram22], using prompt “A teddy bear on a skateboard in times square". We show two versions of our model, one without truncation and the other with truncation. Our model enjoys faster speed than the diffusion models in both cases. Still, we observe our model falls behind in details, like the exact shape of skateboards. For LDM and Stable Diffusion, we use 250 and 50 sampling steps with DDIM / PLMS [Liu22h], respectively. For DALL$\cdot$E 2, we generate images using the official DALL$\cdot$E service [Dal22].
<span id="figure-22"></span>

![Figure A13](../../papers/gigagan/figure-22.png)

**Figure A13.** Random outputs of our model, Latent Diffusion Model [Rom22], Stable Diffusion [Sta22], and DALL$\cdot$E 2 [Ram22], using prompt “Vibrant portrait painting of Salvador Dalí with a robotic half face". We show two versions of our model, one without truncation and the other with truncation. Our model enjoys faster speed than the diffusion models in both cases. Still, we observe our model falls behind in structural details like in the detailed shape of eyes. For LDM and Stable Diffusion, we use 250 and 50 sampling steps with DDIM / PLMS [Liu22h], respectively. For DALL$\cdot$E 2, we generate images using the official DALL$\cdot$E service [Dal22].
<span id="figure-23"></span>

![Figure A14](../../papers/gigagan/figure-23.png)

**Figure A14.** Random outputs of our model, Latent Diffusion Model [Rom22], Stable Diffusion [Sta22], and DALL$\cdot$E 2 [Ram22], using prompt “Three men in military suits are sitting on a bench". We show two versions of our model, one without truncation and the other with truncation. Our model enjoys faster speed than the diffusion models in both cases. Still, we observe our model falls behind in details in facial expression and attire. For LDM and Stable Diffusion, we use 250 and 50 sampling steps with DDIM / PLMS [Liu22h], respectively. For DALL$\cdot$E 2, we generate images using the official DALL$\cdot$E service [Dal22].
<span id="figure-24"></span>

![Figure A15](../../papers/gigagan/figure-24.png)

**Figure A15.** Our GAN-based upsampler can serve as the upsampler for many text-to-image models that generate initial outputs at low resolutions like 64px or 128px. We simulate such usage by applying our 8$\times$ superresolution model on a low-res 128px artwork to obtain the 1K output, using “Portrait of a kitten dressed in a bow tie. Red Rose. Valentine’s day.". Then our model can be re-applied to go beyond 4K. We compare our model with the text-conditioned upscaler of Stable Diffusion [Sta22] and unconditional Real-ESRGAN [Wan21i]. Zooming in is recommended for comparison between 1K and 4K outputs.
<span id="figure-25"></span>

![Figure A16](../../papers/gigagan/figure-25.png)

**Figure A16.** Our GAN-based upsampler can serve as the upsampler for many text-to-image models that generate initial outputs at low resolutions like 64px or 128px. We simulate such usage by applying our 8$\times$ superresolution model on a low-res 128px artwork to obtain the 1K output, using “Heart shaped pancakes with honey and strawberry for Valentine’s Day". Then our model can be re-applied to go beyond 4K. We compare our model with the text-conditioned upscaler of Stable Diffusion [Sta22] and unconditional Real-ESRGAN [Wan21i]. Zooming in is recommended for comparison between 1K and 4K outputs.
<span id="figure-26"></span>

![Figure A17](../../papers/gigagan/figure-26.png)

**Figure A17.** Our GAN-based upsampler can also be used as an off-the-shelf superresolution model for real images with a large scaling factor by providing an appropriate description of the image. We apply our text-conditioned 8$\times$ superresolution model on a low-res 128px photo to obtain the 1K output, using “An elephant spraying water with its trunk". Then our model can be re-applied to go beyond 4K. We compare our model with the text-conditioned upscaler of Stable Diffusion [Sta22] and unconditional Real-ESRGAN [Wan21i]. Zooming in is recommended for comparison between 1K and 4K outputs.
