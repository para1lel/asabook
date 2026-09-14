---
title: Consistency Models
createTime: 2026/09/14 11:58:45
permalink: /en/papers/consistency-models/
pageClass: paper-reading
---

> [Yang Song](https://yang-song.net/), [Prafulla Dhariwal](https://dblp.org/pid/190/7235), [Mark Chen](https://dblp.org/pid/40/1660-3), and [Ilya Sutskever](https://dblp.org/pid/60/5276). First submitted to arXiv on March 2, 2023; current version v2. Published in the [Proceedings of the 40th International Conference on Machine Learning](https://proceedings.mlr.press/v202/song23a.html), PMLR 202:32211-32252, 2023. [Consistency Models](https://arxiv.org/abs/2303.01469). <a href="/paper/consistency-models.pdf" target="_blank" rel="noopener noreferrer">Original PDF</a>. [DOI](https://doi.org/10.48550/arXiv.2303.01469). [TeX source](https://export.arxiv.org/e-print/2303.01469v2). The original PDF remains authoritative for the exact print layout and bibliography.

## Abstract

Diffusion models have significantly advanced the fields of image, audio, and video generation, but they depend on an iterative sampling process that causes slow generation. To overcome this limitation, we propose *consistency models*, a new family of models that generate high quality samples by directly mapping noise to data. They support fast one-step generation by design, while still allowing multistep sampling to trade compute for sample quality. They also support zero-shot data editing, such as image inpainting, colorization, and super-resolution, without requiring explicit training on these tasks. Consistency models can be trained either by distilling pre-trained diffusion models, or as standalone generative models altogether. Through extensive experiments, we demonstrate that they outperform existing distillation techniques for diffusion models in one- and few-step sampling, achieving the new state-of-the-art FID of 3.55 on CIFAR-10 and 6.20 on ImageNet $64\times 64$ for one-step generation. When trained in isolation, consistency models become a new family of generative models that can outperform existing one-step, non-adversarial generative models on standard benchmarks such as CIFAR-10, ImageNet $64\times 64$ and LSUN $256\times 256$.

<span id="section-1"></span>

## 1 Introduction

<span id="figure-01"></span>

![Figure 1. Given a Probability Flow (PF) ODE that smoothly converts data to noise, we learn to map any point (*e.g*., ${\mathbf{x}}_{t}$, ${\mathbf{x}}_{t^{\prime}}$, and ${\mathbf{x}}_{T}$) on the ODE trajectory to its origin (*e.g*., ${\mathbf{x}}_{0}$) for generative modeling. Models of these mappings are called consistency models, as their outputs are trained to be consistent for points on the same trajectory.](../../papers/consistency-models/figure-01.png)

**Figure 1.** Given a Probability Flow (PF) ODE that smoothly converts data to noise, we learn to map any point (*e.g*., ${\mathbf{x}}_{t}$, ${\mathbf{x}}_{t^{\prime}}$, and ${\mathbf{x}}_{T}$) on the ODE trajectory to its origin (*e.g*., ${\mathbf{x}}_{0}$) for generative modeling. Models of these mappings are called consistency models, as their outputs are trained to be consistent for points on the same trajectory.

Diffusion models [Soh15, Son19a, Son20, Nic21, Son21], also known as score-based generative models, have achieved unprecedented success across multiple fields, including image generation [Dha21, Nic22, Ram22, Sah22a, Rom22], audio synthesis [Kon21, Che21n, Pop21], and video generation [Ho22, Ho22c]. A key feature of diffusion models is the iterative sampling process which progressively removes noise from random initial vectors. This iterative process provides a flexible trade-off of compute and sample quality, as using extra compute for more iterations usually yields samples of better quality. It is also the crux of many zero-shot data editing capabilities of diffusion models, enabling them to solve challenging inverse problems ranging from image inpainting, colorization, stroke-guided image editing, to Computed Tomography and Magnetic Resonance Imaging [Son19a, Son21, Son22, Son23a, Kaw21, Kaw22, Chu23, Men22]. However, compared to single-step generative models like GANs [Goo14a], VAEs [Kin14, Rez14], or normalizing flows [Din15, Din17, Kin18], the iterative generation procedure of diffusion models typically requires 10–2000 times more compute for sample generation [Son20, Nic21, Son21, Zha22i, Lu22c], causing slow inference and limited real-time applications.

Our objective is to create generative models that facilitate efficient, single-step generation without sacrificing important advantages of iterative sampling, such as trading compute for sample quality when necessary, as well as performing zero-shot data editing tasks. As illustrated in [Figure 1](#figure-01), we build on top of the probability flow (PF) ordinary differential equation (ODE) in continuous-time diffusion models [Son21], whose trajectories smoothly transition the data distribution into a tractable noise distribution. We propose to learn a model that maps any point at any time step to the trajectory’s starting point. A notable property of our model is self-consistency: *points on the same trajectory map to the same initial point*. We therefore refer to such models as **consistency models**. Consistency models allow us to generate data samples (initial points of ODE trajectories, *e.g*., ${\mathbf{x}}_{0}$ in [Figure 1](#figure-01)) by converting random noise vectors (endpoints of ODE trajectories, *e.g*., ${\mathbf{x}}_{T}$ in [Figure 1](#figure-01)) with only one network evaluation. Importantly, by chaining the outputs of consistency models at multiple time steps, we can improve sample quality and perform zero-shot data editing at the cost of more compute, similar to what iterative sampling enables for diffusion models.

To train a consistency model, we offer two methods based on enforcing the self-consistency property. The first method relies on using numerical ODE solvers and a pre-trained diffusion model to generate pairs of adjacent points on a PF ODE trajectory. By minimizing the difference between model outputs for these pairs, we can effectively distill a diffusion model into a consistency model, which allows generating high-quality samples with one network evaluation. By contrast, our second method eliminates the need for a pre-trained diffusion model altogether, allowing us to train a consistency model in isolation. This approach situates consistency models as an independent family of generative models. Importantly, neither approach necessitates adversarial training, and they both place minor constraints on the architecture, allowing the use of flexible neural networks for parameterizing consistency models.

We demonstrate the efficacy of consistency models on several image datasets, including CIFAR-10 [Kri09], ImageNet $64\times 64$ [Den09a], and LSUN $256\times 256$ [Yu15a]. Empirically, we observe that as a distillation approach, consistency models outperform existing diffusion distillation methods like progressive distillation [Sal22] across a variety of datasets in few-step generation: On CIFAR-10, consistency models reach new state-of-the-art FIDs of 3.55 and 2.93 for one-step and two-step generation; on ImageNet $64\times 64$, it achieves record-breaking FIDs of 6.20 and 4.70 with one and two network evaluations respectively. When trained as standalone generative models, consistency models can match or surpass the quality of one-step samples from progressive distillation, despite having no access to pre-trained diffusion models. They are also able to outperform many GANs, and existing non-adversarial, single-step generative models across multiple datasets. Furthermore, we show that consistency models can be used to perform a wide range of zero-shot data editing tasks, including image denoising, interpolation, inpainting, colorization, super-resolution, and stroke-guided image editing (SDEdit, [Men22]).

<span id="section-2"></span>

## 2 Diffusion Models

Consistency models are heavily inspired by the theory of continuous-time diffusion models [Son21, Kar22]. Diffusion models generate data by progressively perturbing data to noise via Gaussian perturbations, then creating samples from noise via sequential denoising steps. Let $p_{\text{data}}({\mathbf{x}})$ denote the data distribution. Diffusion models start by diffusing $p_{\text{data}}({\mathbf{x}})$ with a stochastic differential equation (SDE) [Son21]

<span id="equation-01"></span>

$$
\mathop{}\!\mathrm{d}{\mathbf{x}}_{t}=\bm{\mu}({\mathbf{x}}_{t},t)\mathop{}\!\mathrm{d}t+\sigma(t)\mathop{}\!\mathrm{d}{\mathbf{w}}_{t},
$$

where $t\in[0,T]$, $T>0$ is a fixed constant, $\bm{\mu}(\cdot,\cdot)$ and $\sigma(\cdot)$ are the drift and diffusion coefficients respectively, and $\{{\mathbf{w}}_{t}\}_{t\in[0,T]}$ denotes the standard Brownian motion. We denote the distribution of ${\mathbf{x}}_{t}$ as $p_{t}({\mathbf{x}})$ and as a result $p_{0}({\mathbf{x}})\equiv p_{\text{data}}({\mathbf{x}})$. A remarkable property of this SDE is the existence of an ordinary differential equation (ODE), dubbed the *Probability Flow (PF) ODE* by [Son21], whose solution trajectories sampled at $t$ are distributed according to $p_{t}({\mathbf{x}})$:

<span id="equation-02"></span>

$$
\mathop{}\!\mathrm{d}{\mathbf{x}}_{t}=\left[\bm{\mu}({\mathbf{x}}_{t},t)-\frac{1}{2}\sigma(t)^{2}\nabla\log p_{t}({\mathbf{x}}_{t})\right]\mathop{}\!\mathrm{d}t.
$$

Here $\nabla\log p_{t}({\mathbf{x}})$ is the *score function* of $p_{t}({\mathbf{x}})$; hence diffusion models are also known as *score-based generative models* [Son19a, Son20, Son21].

Typically, the SDE in [Equation 1](#equation-01) is designed such that $p_{T}({\mathbf{x}})$ is close to a tractable Gaussian distribution $\pi({\mathbf{x}})$. We hereafter adopt the settings in [Kar22], where $\bm{\mu}({\mathbf{x}},t)=\bm{0}$ and $\sigma(t)=\sqrt{2t}$. In this case, we have $p_{t}({\mathbf{x}})=p_{\text{data}}({\mathbf{x}})\otimes\mathcal{N}(\bm{0},t^{2}{\bm{I}})$, where $\otimes$ denotes the convolution operation, and $\pi({\mathbf{x}})=\mathcal{N}(\bm{0},T^{2}{\bm{I}})$. For sampling, we first train a *score model* ${\bm{s}}_{\bm{\phi}}({\mathbf{x}},t)\approx\nabla\log p_{t}({\mathbf{x}})$ via *score matching* [Hyv05, Vin11, Son19b, Son19a, Nic21], then plug it into [Equation 2](#equation-02) to obtain an empirical estimate of the PF ODE, which takes the form of

<span id="equation-03"></span>

$$
\frac{\mathop{}\!\mathrm{d}{\mathbf{x}}_{t}}{\mathop{}\!\mathrm{d}t}=-t{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t},t).
$$

We call [Equation 3](#equation-03) the *empirical PF ODE*. Next, we sample $\hat{{\mathbf{x}}}_{T}\sim\pi=\mathcal{N}(\bm{0},T^{2}{\bm{I}})$ to initialize the empirical PF ODE and solve it backwards in time with any numerical ODE solver, such as Euler [Son21a, Son21] and Heun solvers [Kar22], to obtain the solution trajectory $\{\hat{{\mathbf{x}}}_{t}\}_{t\in[0,T]}$. The resulting $\hat{{\mathbf{x}}}_{0}$ can then be viewed as an approximate sample from the data distribution $p_{\text{data}}({\mathbf{x}})$. To avoid numerical instability, one typically stops the solver at $t=\epsilon$, where $\epsilon$ is a fixed small positive number, and accepts $\hat{{\mathbf{x}}}_{\epsilon}$ as the approximate sample. Following [Kar22], we rescale image pixel values to $[-1,1]$, and set $T=80,\epsilon=0.002$.

Diffusion models are bottlenecked by their slow sampling speed. Clearly, using ODE solvers for sampling requires iterative evaluations of the score model ${\bm{s}}_{\bm{\phi}}({\mathbf{x}},t)$, which is computationally costly. Existing methods for fast sampling include faster numerical ODE solvers [Son21a, Zha22i, Lu22c, Doc22a], and distillation techniques [Luh21, Sal22, Men22a, Zhe22g]. However, ODE solvers still need more than 10 evaluation steps to generate competitive samples. Most distillation methods like [Luh21] and [Zhe22g] rely on collecting a large dataset of samples from the diffusion model prior to distillation, which itself is computationally expensive. To our best knowledge, the only distillation approach that does not suffer from this drawback is progressive distillation (PD, [Sal22]), with which we compare consistency models extensively in our experiments.

<span id="section-3"></span>

## 3 Consistency Models

<span id="figure-02"></span>

![Figure 2. Consistency models are trained to map points on any trajectory of the PF ODE to the trajectory’s origin.](../../papers/consistency-models/figure-02.png)

**Figure 2.** Consistency models are trained to map points on any trajectory of the PF ODE to the trajectory’s origin.

We propose consistency models, a new type of models that support single-step generation at the core of its design, while still allowing iterative generation for trade-offs between sample quality and compute, and zero-shot data editing. Consistency models can be trained in either the distillation mode or the isolation mode. In the former case, consistency models distill the knowledge of pre-trained diffusion models into a single-step sampler, significantly improving other distillation approaches in sample quality, while allowing zero-shot image editing applications. In the latter case, consistency models are trained in isolation, with no dependence on pre-trained diffusion models. This makes them an independent new class of generative models.

Below we introduce the definition, parameterization, and sampling of consistency models, plus a brief discussion on their applications to zero-shot data editing.

**Definition** Given a solution trajectory $\{{\mathbf{x}}_{t}\}_{t\in[\epsilon,T]}$ of the PF ODE in [Equation 2](#equation-02), we define the *consistency function* as ${\bm{f}}:({\mathbf{x}}_{t},t)\mapsto{\mathbf{x}}_{\epsilon}$. A consistency function has the property of *self-consistency*: its outputs are consistent for arbitrary pairs of $({\mathbf{x}}_{t},t)$ that belong to the same PF ODE trajectory, *i.e*., ${\bm{f}}({\mathbf{x}}_{t},t)={\bm{f}}({\mathbf{x}}_{t^{\prime}},t^{\prime})$ for all $t,t^{\prime}\in[\epsilon,T]$. As illustrated in [Figure 2](#figure-02), the goal of a *consistency model*, symbolized as ${\bm{f}}_{\bm{\theta}}$, is to estimate this consistency function ${\bm{f}}$ from data by learning to enforce the self-consistency property (details in [Section 4](#section-4) and [Section 5](#section-5)). Note that a similar definition is used for neural flows [Bil21] in the context of neural ODEs [Che18g]. Compared to neural flows, however, we do not enforce consistency models to be invertible.

**Parameterization** For any consistency function ${\bm{f}}(\cdot,\cdot)$, we have ${\bm{f}}({\mathbf{x}}_{\epsilon},\epsilon)={\mathbf{x}}_{\epsilon}$, *i.e*., ${\bm{f}}(\cdot,\epsilon)$ is an identity function. We call this constraint the *boundary condition*. All consistency models have to meet this boundary condition, as it plays a crucial role in the successful training of consistency models. This boundary condition is also the most confining architectural constraint on consistency models. For consistency models based on deep neural networks, we discuss two ways to implement this boundary condition *almost for free*. Suppose we have a free-form deep neural network $F_{\bm{\theta}}({\mathbf{x}},t)$ whose output has the same dimensionality as ${\mathbf{x}}$. The first way is to simply parameterize the consistency model as

<span id="equation-04"></span>

$$
{\bm{f}}_{\bm{\theta}}({\mathbf{x}},t)=\begin{cases}{\mathbf{x}}&\quad t=\epsilon\\
F_{\bm{\theta}}({\mathbf{x}},t)&\quad t\in(\epsilon,T]\end{cases}.
$$

The second method is to parameterize the consistency model using skip connections, that is,

<span id="equation-05"></span>

$$
{\bm{f}}_{\bm{\theta}}({\mathbf{x}},t)=c_{\text{skip}}(t){\mathbf{x}}+c_{\text{out}}(t)F_{\bm{\theta}}({\mathbf{x}},t),
$$

where $c_{\text{skip}}(t)$ and $c_{\text{out}}(t)$ are differentiable functions such that $c_{\text{skip}}(\epsilon)=1$, and $c_{\text{out}}(\epsilon)=0$. This way, the consistency model is differentiable at $t=\epsilon$ if $F_{\bm{\theta}}({\mathbf{x}},t),c_{\text{skip}}(t),c_{\text{out}}(t)$ are all differentiable, which is critical for training continuous-time consistency models ([Section 9.1](#section-9-1) and [Section 9.2](#section-9-2)). The parameterization in [Equation 5](#equation-05) bears strong resemblance to many successful diffusion models [Kar22, Bal22], making it easier to borrow powerful diffusion model architectures for constructing consistency models. We therefore follow the second parameterization in all experiments.

**Sampling** With a well-trained consistency model ${\bm{f}}_{\bm{\theta}}(\cdot,\cdot)$, we can generate samples by sampling from the initial distribution $\hat{{\mathbf{x}}}_{T}\sim\mathcal{N}(\bm{0},T^{2}{\bm{I}})$ and then evaluating the consistency model for $\hat{{\mathbf{x}}}_{\epsilon}={\bm{f}}_{\bm{\theta}}(\hat{{\mathbf{x}}}_{T},T)$. This involves only one forward pass through the consistency model and therefore *generates samples in a single step*. Importantly, one can also evaluate the consistency model multiple times by alternating denoising and noise injection steps for improved sample quality. Summarized in [Algorithm 1](#algorithm-01), this *multistep* sampling procedure provides the flexibility to trade compute for sample quality. It also has important applications in zero-shot data editing. In practice, we find time points $\{\tau_{1},\tau_{2},\cdots,\tau_{N-1}\}$ in [Algorithm 1](#algorithm-01) with a greedy algorithm, where the time points are pinpointed one at a time using ternary search to optimize the FID of samples obtained from [Algorithm 1](#algorithm-01). This assumes that given prior time points, the FID is a unimodal function of the next time point. We find this assumption to hold empirically in our experiments, and leave the exploration of better strategies as future work.

<span id="algorithm-01"></span>

<div class="paper-algorithm">

**Algorithm 1: Multistep Consistency Sampling.**

- **Input:** Consistency model ${\bm{f}}_{\bm{\theta}}(\cdot,\cdot)$, sequence of time points $\tau_{1}>\tau_{2}>\cdots>\tau_{N-1}$, initial noise $\hat{{\mathbf{x}}}_{T}$.
- ${\mathbf{x}}\gets{\bm{f}}_{\bm{\theta}}(\hat{{\mathbf{x}}}_{T},T)$.
- **For** $n=1$ **to** $N-1$:
  - Sample ${\mathbf{z}}\sim\mathcal{N}(\bm{0},{\bm{I}})$.
  - $\hat{{\mathbf{x}}}_{\tau_{n}}\gets{\mathbf{x}}+\sqrt{\tau_{n}^{2}-\epsilon^{2}}{\mathbf{z}}$.
  - ${\mathbf{x}}\gets{\bm{f}}_{\bm{\theta}}(\hat{{\mathbf{x}}}_{\tau_{n}},\tau_{n})$.
- **Output:** ${\mathbf{x}}$.

</div>

**Zero-Shot Data Editing** Similar to diffusion models, consistency models enable various data editing and manipulation applications in zero shot; they do not require explicit training to perform these tasks. For example, consistency models define a one-to-one mapping from a Gaussian noise vector to a data sample. Similar to latent variable models like GANs, VAEs, and normalizing flows, consistency models can easily interpolate between samples by traversing the latent space ([Figure 11](#figure-11)). As consistency models are trained to recover ${\mathbf{x}}_{\epsilon}$ from any noisy input ${\mathbf{x}}_{t}$ where $t\in[\epsilon,T]$, they can perform denoising for various noise levels ([Figure 12](#figure-12)). Moreover, the multistep generation procedure in [Algorithm 1](#algorithm-01) is useful for solving certain inverse problems in zero shot by using an iterative replacement procedure similar to that of diffusion models [Son19a, Son21, Ho22]. This enables many applications in the context of image editing, including inpainting ([Figure 10](#figure-10)), colorization ([Figure 8](#figure-08)), super-resolution ([Figure 6(b)](#figure-06)) and stroke-guided image editing ([Figure 13](#figure-13)) as in SDEdit [Men22]. In [Section 6.3](#section-6-3), we empirically demonstrate the power of consistency models on many zero-shot image editing tasks.

<span id="section-4"></span>

## 4 Training Consistency Models via Distillation

We present our first method for training consistency models based on distilling a pre-trained score model ${\bm{s}}_{\bm{\phi}}({\mathbf{x}},t)$. Our discussion revolves around the empirical PF ODE in [Equation 3](#equation-03), obtained by plugging the score model ${\bm{s}}_{\bm{\phi}}({\mathbf{x}},t)$ into the PF ODE. Consider discretizing the time horizon $[\epsilon,T]$ into $N-1$ sub-intervals, with boundaries $t_{1}=\epsilon<t_{2}<\cdots<t_{N}=T$. In practice, we follow [Kar22] to determine the boundaries with the formula $t_{i}=(\epsilon^{1/\rho}+\frac{i-1}{N-1}(T^{1/\rho}-\epsilon^{1/\rho}))^{\rho}$, where $\rho=7$. When $N$ is sufficiently large, we can obtain an accurate estimate of ${\mathbf{x}}_{t_{n}}$ from ${\mathbf{x}}_{t_{n+1}}$ by running one discretization step of a numerical ODE solver. This estimate, which we denote as $\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}}$, is defined by

<span id="equation-06"></span>

$$
\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}}\coloneqq{\mathbf{x}}_{t_{n+1}}+(t_{n}-t_{n+1})\Phi({\mathbf{x}}_{t_{n+1}},t_{n+1};{\bm{\phi}}),
$$

where $\Phi(\cdots;{\bm{\phi}})$ represents the update function of a one-step ODE solver applied to the empirical PF ODE. For example, when using the Euler solver, we have $\Phi({\mathbf{x}},t;{\bm{\phi}})=-t{\bm{s}}_{\bm{\phi}}({\mathbf{x}},t)$ which corresponds to the following update rule

$$
\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}}={\mathbf{x}}_{t_{n+1}}-(t_{n}-t_{n+1})t_{n+1}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t_{n+1}},t_{n+1}).
$$

For simplicity, we only consider one-step ODE solvers in this work. It is straightforward to generalize our framework to multistep ODE solvers and we leave it as future work.

Due to the connection between the PF ODE in [Equation 2](#equation-02) and the SDE in [Equation 1](#equation-01) (see [Section 2](#section-2)), one can sample along the distribution of ODE trajectories by first sampling ${\mathbf{x}}\sim p_{\text{data}}$, then adding Gaussian noise to ${\mathbf{x}}$. Specifically, given a data point ${\mathbf{x}}$, we can generate a pair of adjacent data points $(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},{\mathbf{x}}_{t_{n+1}})$ on the PF ODE trajectory efficiently by sampling ${\mathbf{x}}$ from the dataset, followed by sampling ${\mathbf{x}}_{t_{n+1}}$ from the transition density of the SDE $\mathcal{N}({\mathbf{x}},t_{n+1}^{2}{\bm{I}})$, and then computing $\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}}$ using one discretization step of the numerical ODE solver according to [Equation 6](#equation-06). Afterwards, we train the consistency model by minimizing its output differences on the pair $(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},{\mathbf{x}}_{t_{n+1}})$. This motivates our following *consistency distillation* loss for training consistency models.

<span id="definition-01"></span>

**Definition 1.** The consistency distillation loss is defined as

<span id="equation-07"></span>

$$
\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})\coloneqq\\
\mathbb{E}[\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({{\mathbf{x}}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))],
$$

where the expectation is taken with respect to ${\mathbf{x}}\sim p_{\text{data}}$, $n\sim\mathcal{U}\llbracket 1,N-1\rrbracket$, and ${\mathbf{x}}_{t_{n+1}}\sim\mathcal{N}({\mathbf{x}};t_{n+1}^{2}{\bm{I}})$. Here $\mathcal{U}\llbracket 1,N-1\rrbracket$ denotes the uniform distribution over $\{1,2,\cdots,N-1\}$, $\lambda(\cdot)\in\mathbb{R}^{+}$ is a positive weighting function, $\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}}$ is given by [Equation 6](#equation-06), ${\bm{\theta}}^{-}$ denotes a running average of the past values of ${\bm{\theta}}$ during the course of optimization, and $d(\cdot,\cdot)$ is a metric function that satisfies $\forall{\mathbf{x}},{\mathbf{y}}:d({\mathbf{x}},{\mathbf{y}})\geq 0$ and $d({\mathbf{x}},{\mathbf{y}})=0$ if and only if ${\mathbf{x}}={\mathbf{y}}$.

Unless otherwise stated, we adopt the notations in [Definition 1](#definition-01) throughout this paper, and use $\mathbb{E}[\cdot]$ to denote the expectation over all random variables. In our experiments, we consider the squared $\ell_{2}$ distance $d({\mathbf{x}},{\mathbf{y}})=\|{\mathbf{x}}-{\mathbf{y}}\|^{2}_{2}$, $\ell_{1}$ distance $d({\mathbf{x}},{\mathbf{y}})=\|{\mathbf{x}}-{\mathbf{y}}\|_{1}$, and the Learned Perceptual Image Patch Similarity (LPIPS, [Zha18d]). We find $\lambda(t_{n})\equiv 1$ performs well across all tasks and datasets. In practice, we minimize the objective by stochastic gradient descent on the model parameters ${\bm{\theta}}$, while updating ${\bm{\theta}}^{-}$ with exponential moving average (EMA). That is, given a decay rate $0\leq\mu<1$, we perform the following update after each optimization step:

<span id="equation-08"></span>

$$
{\bm{\theta}}^{-}\leftarrow\operatorname{stopgrad}(\mu{\bm{\theta}}^{-}+(1-\mu){\bm{\theta}}).
$$

The overall training procedure is summarized in [Algorithm 2](#algorithm-02). In alignment with the convention in deep reinforcement learning [Mni13, Mni15, Lil15] and momentum based contrastive learning [Gri20, He20a], we refer to ${\bm{f}}_{{\bm{\theta}}^{-}}$ as the “target network”, and ${\bm{f}}_{\bm{\theta}}$ as the “online network”. We find that compared to simply setting ${\bm{\theta}}^{-}={\bm{\theta}}$, the EMA update and “stopgrad” operator in [Equation 8](#equation-08) can greatly stabilize the training process and improve the final performance of the consistency model.

<span id="algorithm-02"></span>

<div class="paper-algorithm">

**Algorithm 2: Consistency Distillation (CD).**

- **Input:** Dataset $\mathcal{D}$, initial model parameter $\bm{\theta}$, learning rate $\eta$, ODE solver $\Phi(\cdot,\cdot;\bm{\phi})$, $d(\cdot,\cdot)$, $\lambda(\cdot)$, and $\mu$.
- $\bm{\theta}^{-}\gets\bm{\theta}$.
- **Repeat until** convergence:
  - Sample ${\mathbf{x}}\sim\mathcal{D}$ and $n\sim\mathcal{U}\llbracket 1,N-1\rrbracket$.
  - Sample ${\mathbf{x}}_{t_{n+1}}\sim\mathcal{N}({\mathbf{x}};t_{n+1}^{2}{\bm{I}})$.
  - $\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}}\gets{\mathbf{x}}_{t_{n+1}}+(t_{n}-t_{n+1})\Phi({\mathbf{x}}_{t_{n+1}},t_{n+1};\bm{\phi})$.
  - $\mathcal{L}(\bm{\theta},\bm{\theta}^{-};\bm{\phi})\gets\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{\bm{\theta}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))$.
  - $\bm{\theta}\gets\bm{\theta}-\eta\nabla_{\bm{\theta}}\mathcal{L}(\bm{\theta},\bm{\theta}^{-};\bm{\phi})$.
  - $\bm{\theta}^{-}\gets\operatorname{stopgrad}(\mu\bm{\theta}^{-}+(1-\mu)\bm{\theta})$.

</div>

Below we provide a theoretical justification for consistency distillation based on asymptotic analysis.

<span id="theorem-01"></span>

**Theorem 1.** Let $\Delta t\coloneqq\max_{n\in\llbracket 1,N-1\rrbracket}\{|t_{n+1}-t_{n}|\}$, and ${\bm{f}}(\cdot,\cdot;{\bm{\phi}})$ be the consistency function of the empirical PF ODE in [Equation 3](#equation-03). Assume ${\bm{f}}_{\bm{\theta}}$ satisfies the Lipschitz condition: there exists $L>0$ such that for all $t\in[\epsilon,T]$, ${\mathbf{x}}$, and ${\mathbf{y}}$, we have $\|{\bm{f}}_{\bm{\theta}}({\mathbf{x}},t)-{\bm{f}}_{\bm{\theta}}({\mathbf{y}},t)\|_{2}\leq L\|{\mathbf{x}}-{\mathbf{y}}\|_{2}$. Assume further that for all $n\in\llbracket 1,N-1\rrbracket$, the ODE solver called at $t_{n+1}$ has local error uniformly bounded by $O((t_{n+1}-t_{n})^{p+1})$ with $p\geq 1$. Then, if $\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}};{\bm{\phi}})=0$, we have

$$
\sup_{n,{\mathbf{x}}}\|{\bm{f}}_{{\bm{\theta}}}({\mathbf{x}},t_{n})-{\bm{f}}({\mathbf{x}},t_{n};{\bm{\phi}})\|_{2}=O((\Delta t)^{p}).
$$

::: details Proof
The proof is based on induction and parallels the classic proof of global error bounds for numerical ODE solvers [Sul03]. We provide the full proof in [Section 8.2](#section-8-2).
:::

Since ${\bm{\theta}}^{-}$ is a running average of the history of ${\bm{\theta}}$, we have ${\bm{\theta}}^{-}={\bm{\theta}}$ when the optimization of [Algorithm 2](#algorithm-02) converges. That is, the target and online consistency models will eventually match each other. If the consistency model additionally achieves zero consistency distillation loss, then [Theorem 1](#theorem-01) implies that, under some regularity conditions, the estimated consistency model can become arbitrarily accurate, as long as the step size of the ODE solver is sufficiently small. Importantly, our boundary condition ${\bm{f}}_{\bm{\theta}}({\mathbf{x}},\epsilon)\equiv{\mathbf{x}}$ precludes the trivial solution ${\bm{f}}_{\bm{\theta}}({\mathbf{x}},t)\equiv\bm{0}$ from arising in consistency model training.

The consistency distillation loss $\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})$ can be extended to hold for infinitely many time steps ($N\to\infty$) if ${\bm{\theta}}^{-}={\bm{\theta}}$ or ${\bm{\theta}}^{-}=\operatorname{stopgrad}({\bm{\theta}})$. The resulting continuous-time loss functions do not require specifying $N$ nor the time steps $\{t_{1},t_{2},\cdots,t_{N}\}$. Nonetheless, they involve Jacobian-vector products and require forward-mode automatic differentiation for efficient implementation, which may not be well-supported in some deep learning frameworks. We provide these continuous-time distillation loss functions in Theorems 3, 4 and 5, and relegate details to [Section 9.1](#section-9-1).

<span id="section-5"></span>

## 5 Training Consistency Models in Isolation

Consistency models can be trained without relying on any pre-trained diffusion models. This differs from existing diffusion distillation techniques, making consistency models a new independent family of generative models.

<span id="algorithm-03"></span>

<div class="paper-algorithm">

**Algorithm 3: Consistency Training (CT).**

- **Input:** Dataset $\mathcal{D}$, initial model parameter $\bm{\theta}$, learning rate $\eta$, step schedule $N(\cdot)$, EMA decay rate schedule $\mu(\cdot)$, $d(\cdot,\cdot)$, and $\lambda(\cdot)$.
- $\bm{\theta}^{-}\gets\bm{\theta}$ and $k\gets 0$.
- **Repeat until** convergence:
  - Sample ${\mathbf{x}}\sim\mathcal{D}$, and $n\sim\mathcal{U}\llbracket 1,N(k)-1\rrbracket$.
  - Sample ${\mathbf{z}}\sim\mathcal{N}(\bm{0},{\bm{I}})$.
  - $\mathcal{L}(\bm{\theta},\bm{\theta}^{-})\gets\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1}),{\bm{f}}_{\bm{\theta}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n}))$.
  - $\bm{\theta}\gets\bm{\theta}-\eta\nabla_{\bm{\theta}}\mathcal{L}(\bm{\theta},\bm{\theta}^{-})$.
  - $\bm{\theta}^{-}\gets\operatorname{stopgrad}(\mu(k)\bm{\theta}^{-}+(1-\mu(k))\bm{\theta})$.
  - $k\gets k+1$.

</div>

Recall that in consistency distillation, we rely on a pre-trained score model ${\bm{s}}_{\bm{\phi}}({\mathbf{x}},t)$ to approximate the ground truth score function $\nabla\log p_{t}({\mathbf{x}})$. It turns out that we can avoid this pre-trained score model altogether by leveraging the following unbiased estimator ([Lemma 1](#lemma-01) in [Section 8](#section-8)):

$$
\nabla\log p_{t}({\mathbf{x}}_{t})=-\mathbb{E}\left[\frac{{\mathbf{x}}_{t}-{\mathbf{x}}}{t^{2}}\mathrel{\bigg|}{\mathbf{x}}_{t}\right],
$$

where ${\mathbf{x}}\sim p_{\text{data}}$ and ${\mathbf{x}}_{t}\sim\mathcal{N}({\mathbf{x}};t^{2}{\bm{I}})$. That is, given ${\mathbf{x}}$ and ${\mathbf{x}}_{t}$, we can estimate $\nabla\log p_{t}({\mathbf{x}}_{t})$ with $-({\mathbf{x}}_{t}-{\mathbf{x}})/t^{2}$.

This unbiased estimate suffices to replace the pre-trained diffusion model in consistency distillation when using the Euler method as the ODE solver in the limit of $N\to\infty$, as justified by the following result.

<span id="theorem-02"></span>

**Theorem 2.** Let $\Delta t\coloneqq\max_{n\in\llbracket 1,N-1\rrbracket}\{|t_{n+1}-t_{n}|\}$. Assume $d$ and ${\bm{f}}_{{\bm{\theta}}^{-}}$ are both twice continuously differentiable with bounded second derivatives, the weighting function $\lambda(\cdot)$ is bounded, and $\mathbb{E}[\|\nabla\log p_{t_{n}}({\mathbf{x}}_{t_{n}})\|_{2}^{2}]<\infty$. Assume further that we use the Euler ODE solver, and the pre-trained score model matches the ground truth, *i.e*., $\forall t\in[\epsilon,T]:{\bm{s}}_{{\bm{\phi}}}({\mathbf{x}},t)\equiv\nabla\log p_{t}({\mathbf{x}})$. Then,

<span id="equation-09"></span>

$$
\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})=\mathcal{L}_{\text{CT}}^{N}({\bm{\theta}},{\bm{\theta}}^{-})+o(\Delta t),
$$

where the expectation is taken with respect to ${\mathbf{x}}\sim p_{\text{data}}$, $n\sim\mathcal{U}\llbracket 1,N-1\rrbracket$, and ${\mathbf{x}}_{t_{n+1}}\sim\mathcal{N}({\mathbf{x}};t_{n+1}^{2}{\bm{I}})$. The consistency training objective, denoted by $\mathcal{L}_{\text{CT}}^{N}({\bm{\theta}},{\bm{\theta}}^{-})$, is defined as

<span id="equation-10"></span>

$$
\mathbb{E}[\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n}))],
$$

where ${\mathbf{z}}\sim\mathcal{N}(\bm{0},{\bm{I}})$. Moreover, $\mathcal{L}_{\text{CT}}^{N}({\bm{\theta}},{\bm{\theta}}^{-})\geq O(\Delta t)$ if $\inf_{N}\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})>0$.

::: details Proof
The proof is based on Taylor series expansion and properties of score functions ([Lemma 1](#lemma-01)). A complete proof is provided in [Section 8.3](#section-8-3).
:::

We refer to [Equation 10](#equation-10) as the *consistency training* (CT) loss. Crucially, $\mathcal{L}({\bm{\theta}},{\bm{\theta}}^{-})$ only depends on the online network ${\bm{f}}_{\bm{\theta}}$, and the target network ${\bm{f}}_{{\bm{\theta}}^{-}}$, while being completely agnostic to diffusion model parameters ${\bm{\phi}}$. The loss function $\mathcal{L}({\bm{\theta}},{\bm{\theta}}^{-})\geq O(\Delta t)$ decreases at a slower rate than the remainder $o(\Delta t)$ and thus will dominate the loss in [Equation 9](#equation-09) as $N\to\infty$ and $\Delta t\to 0$.

For improved practical performance, we propose to progressively increase $N$ during training according to a schedule function $N(\cdot)$. The intuition (*cf*., [Figure 3(d)](#figure-03)) is that the consistency training loss has less “variance” but more “bias” with respect to the underlying consistency distillation loss (*i.e*., the left-hand side of [Equation 9](#equation-09)) when $N$ is small (*i.e*., $\Delta t$ is large), which facilitates faster convergence at the beginning of training. On the contrary, it has more “variance” but less “bias” when $N$ is large (*i.e*., $\Delta t$ is small), which is desirable when closer to the end of training. For best performance, we also find that $\mu$ should change along with $N$, according to a schedule function $\mu(\cdot)$. The full algorithm of consistency training is provided in [Algorithm 3](#algorithm-03), and the schedule functions used in our experiments are given in [Section 10](#section-10).

Similar to consistency distillation, the consistency training loss $\mathcal{L}_{\text{CT}}^{N}({\bm{\theta}},{\bm{\theta}}^{-})$ can be extended to hold in continuous time (*i.e*., $N\to\infty$) if ${\bm{\theta}}^{-}=\operatorname{stopgrad}({\bm{\theta}})$, as shown in [Theorem 6](#theorem-06). This continuous-time loss function does not require schedule functions for $N$ or $\mu$, but requires forward-mode automatic differentiation for efficient implementation. Unlike the discrete-time CT loss, there is no undesirable “bias” associated with the continuous-time objective, as we effectively take $\Delta t\to 0$ in [Theorem 2](#theorem-02). We relegate more details to [Section 9.2](#section-9-2).

<span id="section-6"></span>

## 6 Experiments

We employ consistency distillation and consistency training to learn consistency models on real image datasets, including CIFAR-10 [Kri09], ImageNet $64\times 64$ [Den09a], LSUN Bedroom $256\times 256$, and LSUN Cat $256\times 256$ [Yu15a]. Results are compared according to Fréchet Inception Distance (FID, [Heu17], lower is better), Inception Score (IS, [Sal16], higher is better), Precision (Prec., [Kyn19], higher is better), and Recall (Rec., [Kyn19], higher is better). Additional experimental details are provided in [Section 10](#section-10).

<span id="figure-03"></span>

![Figure 3. Various factors that affect consistency distillation (CD) and consistency training (CT) on CIFAR-10. The best configuration for CD is LPIPS, Heun ODE solver, and $N=18$. Our adaptive schedule functions for $N$ and $\mu$ make CT converge significantly faster than fixing them to be constants during the course of optimization.](../../papers/consistency-models/figure-03.png)

**Figure 3.** Various factors that affect consistency distillation (CD) and consistency training (CT) on CIFAR-10. The best configuration for CD is LPIPS, Heun ODE solver, and $N=18$. Our adaptive schedule functions for $N$ and $\mu$ make CT converge significantly faster than fixing them to be constants during the course of optimization.

<span id="figure-04"></span>

![Figure 4. Multistep image generation with consistency distillation (CD). CD outperforms progressive distillation (PD) across all datasets and sampling steps. The only exception is single-step generation on Bedroom $256\times 256$.](../../papers/consistency-models/figure-04.png)

**Figure 4.** Multistep image generation with consistency distillation (CD). CD outperforms progressive distillation (PD) across all datasets and sampling steps. The only exception is single-step generation on Bedroom $256\times 256$.

<span id="section-6-1"></span>

### 6.1 Training Consistency Models

We perform a series of experiments on CIFAR-10 to understand the effect of various hyperparameters on the performance of consistency models trained by consistency distillation (CD) and consistency training (CT). We first focus on the effect of the metric function $d(\cdot,\cdot)$, the ODE solver, and the number of discretization steps $N$ in CD, then investigate the effect of the schedule functions $N(\cdot)$ and $\mu(\cdot)$ in CT.

To set up our experiments for CD, we consider the squared $\ell_{2}$ distance $d({\mathbf{x}},{\mathbf{y}})=\|{\mathbf{x}}-{\mathbf{y}}\|^{2}_{2}$, $\ell_{1}$ distance $d({\mathbf{x}},{\mathbf{y}})=\|{\mathbf{x}}-{\mathbf{y}}\|_{1}$, and the Learned Perceptual Image Patch Similarity (LPIPS, [Zha18d]) as the metric function. For the ODE solver, we compare Euler’s forward method and Heun’s second order method as detailed in [Kar22]. For the number of discretization steps $N$, we compare $N\in\{9,12,18,36,50,60,80,120\}$. All consistency models trained by CD in our experiments are initialized with the corresponding pre-trained diffusion models, whereas models trained by CT are randomly initialized.

As visualized in [Figure 3(a)](#figure-03), the optimal metric for CD is LPIPS, which outperforms both $\ell_{1}$ and $\ell_{2}$ by a large margin over all training iterations. This is expected as the outputs of consistency models are images on CIFAR-10, and LPIPS is specifically designed for measuring the similarity between natural images. Next, we investigate which ODE solver and which discretization step $N$ work the best for CD. As shown in [Figures 3(b)](#figure-03) and [3(c)](#figure-03), Heun ODE solver and $N=18$ are the best choices. Both are in line with the recommendation of [Kar22] despite the fact that we are training consistency models, not diffusion models. Moreover, [Figure 3(b)](#figure-03) shows that with the same $N$, Heun’s second order solver uniformly outperforms Euler’s first order solver. This corroborates with [Theorem 1](#theorem-01), which states that the optimal consistency models trained by higher order ODE solvers have smaller estimation errors with the same $N$. The results of [Figure 3(c)](#figure-03) also indicate that once $N$ is sufficiently large, the performance of CD becomes insensitive to $N$. Given these insights, we hereafter use LPIPS and Heun ODE solver for CD unless otherwise stated. For $N$ in CD, we follow the suggestions in [Kar22] on CIFAR-10 and ImageNet $64\times 64$. We tune $N$ separately on other datasets (details in [Section 10](#section-10)).

Due to the strong connection between CD and CT, we adopt LPIPS for our CT experiments throughout this paper. Unlike CD, there is no need for using Heun’s second order solver in CT as the loss function does not rely on any particular numerical ODE solver. As demonstrated in [Figure 3(d)](#figure-03), the convergence of CT is highly sensitive to $N$—smaller $N$ leads to faster convergence but worse samples, whereas larger $N$ leads to slower convergence but better samples upon convergence. This matches our analysis in [Section 5](#section-5), and motivates our practical choice of progressively growing $N$ and $\mu$ for CT to balance the trade-off between convergence speed and sample quality. As shown in [Figure 3(d)](#figure-03), adaptive schedules of $N$ and $\mu$ significantly improve the convergence speed and sample quality of CT. In our experiments, we tune the schedules $N(\cdot)$ and $\mu(\cdot)$ separately for images of different resolutions, with more details in [Section 10](#section-10).

<span id="table-01"></span>

![Table 1. Sample quality on CIFAR-10.](../../papers/consistency-models/table-01.png)

**Table 1.** Sample quality on CIFAR-10. $^{\ast}$Methods that require synthetic data construction for distillation.

<span id="table-02"></span>

![Table 2. Sample quality on ImageNet $64\times 64$, and LSUN Bedroom and Cat $256\times 256$.](../../papers/consistency-models/table-02.png)

**Table 2.** Sample quality on ImageNet $64\times 64$, and LSUN Bedroom & Cat $256\times 256$. $^{\dagger}$Distillation techniques.

<span id="section-6-2"></span>

### 6.2 Few-Step Image Generation

**Distillation** In current literature, the most directly comparable approach to our consistency distillation (CD) is progressive distillation (PD, [Sal22]); both are thus far the only distillation approaches that *do not construct synthetic data before distillation*. In stark contrast, other distillation techniques, such as knowledge distillation [Luh21] and DFNO [Zhe22g], have to prepare a large synthetic dataset by generating numerous samples from the diffusion model with expensive numerical ODE/SDE solvers. We perform comprehensive comparison for PD and CD on CIFAR-10, ImageNet $64\times 64$, and LSUN $256\times 256$, with all results reported in [Figure 4](#figure-04). All methods distill from an EDM [Kar22] model that we pre-trained in-house. We note that across all sampling iterations, *using the LPIPS metric uniformly improves PD compared to the squared $\ell_{2}$ distance in the original paper of [Sal22]*. Both PD and CD improve as we take more sampling steps. We find that CD uniformly outperforms PD across all datasets, sampling steps, and metric functions considered, except for single-step generation on Bedroom $256\times 256$, where CD with $\ell_{2}$ slightly underperforms PD with $\ell_{2}$. As shown in [Table 1](#table-01), CD even outperforms distillation approaches that require synthetic dataset construction, such as Knowledge Distillation [Luh21] and DFNO [Zhe22g].

<span id="figure-05"></span>

![Figure 5. Samples generated by EDM (*top*), CT + single-step generation (*middle*), and CT + 2-step generation (*Bottom*). All corresponding images are generated from the same initial noise.](../../papers/consistency-models/figure-05.png)

**Figure 5.** Samples generated by EDM (*top*), CT + single-step generation (*middle*), and CT + 2-step generation (*Bottom*). All corresponding images are generated from the same initial noise.

<span id="figure-06"></span>

![Figure 6. Zero-shot image editing with a consistency model trained by consistency distillation on LSUN Bedroom $256\times 256$.](../../papers/consistency-models/figure-06.png)

**Figure 6.** Zero-shot image editing with a consistency model trained by consistency distillation on LSUN Bedroom $256\times 256$.

**Direct Generation** In [Table 1](#table-01) and [Table 2](#table-02), we compare the sample quality of consistency training (CT) with other generative models using one-step and two-step generation. We also include PD and CD results for reference. Both tables report PD results obtained from the $\ell_{2}$ metric function, as this is the default setting used in the original paper of [Sal22]. For fair comparison, we ensure PD and CD distill the same EDM models. In [Table 1](#table-01) and [Table 2](#table-02), we observe that CT outperforms existing single-step, non-adversarial generative models, *i.e*., VAEs and normalizing flows, by a significant margin on CIFAR-10. Moreover, *CT achieves comparable quality to one-step samples from PD without relying on distillation*. In [Figure 5](#figure-05), we provide EDM samples (top), single-step CT samples (middle), and two-step CT samples (bottom). In [Section 12](#section-12), we show additional samples for both CD and CT in [Figures 14](#figure-14), [15](#figure-15), [16](#figure-16), [17](#figure-17), [18](#figure-18), [19](#figure-19), [20](#figure-20) and [21](#figure-21). Importantly, *all samples obtained from the same initial noise vector share significant structural similarity*, even though CT and EDM models are trained independently from one another. This indicates that CT is less likely to suffer from mode collapse, as EDMs do not.

<span id="section-6-3"></span>

### 6.3 Zero-Shot Image Editing

Similar to diffusion models, consistency models allow zero-shot image editing by modifying the multistep sampling process in [Algorithm 1](#algorithm-01). We demonstrate this capability with a consistency model trained on the LSUN bedroom dataset using consistency distillation. In [Figure 6(a)](#figure-06), we show such a consistency model can colorize gray-scale bedroom images at test time, even though it has never been trained on colorization tasks. In [Figure 6(b)](#figure-06), we show the same consistency model can generate high-resolution images from low-resolution inputs. In [Figure 6(c)](#figure-06), we additionally demonstrate that it can generate images based on stroke inputs created by humans, as in SDEdit for diffusion models [Men22]. Again, this editing capability is zero-shot, as the model has not been trained on stroke inputs. In [Section 11](#section-11), we additionally demonstrate the zero-shot capability of consistency models on inpainting ([Figure 10](#figure-10)), interpolation ([Figure 11](#figure-11)) and denoising ([Figure 12](#figure-12)), with more examples on colorization ([Figure 8](#figure-08)), super-resolution ([Figure 9](#figure-09)) and stroke-guided image generation ([Figure 13](#figure-13)).

<span id="section-7"></span>

## 7 Conclusion

We have introduced consistency models, a type of generative models that are specifically designed to support one-step and few-step generation. We have empirically demonstrated that our consistency distillation method outshines the existing distillation techniques for diffusion models on multiple image benchmarks and small sampling iterations. Furthermore, as a standalone generative model, consistency models generate better samples than existing single-step generation models except for GANs. Similar to diffusion models, they also allow zero-shot image editing applications such as inpainting, colorization, super-resolution, denoising, interpolation, and stroke-guided image generation.

In addition, consistency models share striking similarities with techniques employed in other fields, including deep Q-learning [Mni15] and momentum-based contrastive learning [Gri20, He20a]. This offers exciting prospects for cross-pollination of ideas and methods among these diverse fields.

## Acknowledgements

We thank Alex Nichol for reviewing the manuscript and providing valuable feedback, Chenlin Meng for providing stroke inputs needed in our stroke-guided image generation experiments, and the OpenAI Algorithms team.

<span id="section-8"></span>

## 8 Proofs

<span id="section-8-1"></span>

### 8.1 Notations

We use ${\bm{f}}_{{\bm{\theta}}}({\mathbf{x}},t)$ to denote a consistency model parameterized by ${\bm{\theta}}$, and ${\bm{f}}({\mathbf{x}},t;{\bm{\phi}})$ the consistency function of the empirical PF ODE in [Equation 3](#equation-03). Here ${\bm{\phi}}$ symbolizes its dependency on the pre-trained score model ${\bm{s}}_{\bm{\phi}}({\mathbf{x}},t)$. For the consistency function of the PF ODE in [Equation 2](#equation-02), we denote it as ${\bm{f}}({\mathbf{x}},t)$. Given a multi-variate function ${\bm{h}}({\mathbf{x}},{\mathbf{y}})$, we let $\partial_{1}{\bm{h}}({\mathbf{x}},{\mathbf{y}})$ denote the Jacobian of ${\bm{h}}$ over ${\mathbf{x}}$, and analogously $\partial_{2}{\bm{h}}({\mathbf{x}},{\mathbf{y}})$ denote the Jacobian of ${\bm{h}}$ over ${\mathbf{y}}$. Unless otherwise stated, ${\mathbf{x}}$ is supposed to be a random variable sampled from the data distribution $p_{\text{data}}({\mathbf{x}})$, $n$ is sampled uniformly at random from $\llbracket 1,N-1\rrbracket$, and ${\mathbf{x}}_{t_{n}}$ is sampled from $\mathcal{N}({\mathbf{x}};t_{n}^{2}{\bm{I}})$. Here $\llbracket 1,N-1\rrbracket$ represents the set of integers $\{1,2,\cdots,N-1\}$. Furthermore, recall that we define

$$
\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}}\coloneqq{\mathbf{x}}_{t_{n+1}}+(t_{n}-t_{n+1})\Phi({\mathbf{x}}_{t_{n+1}},t_{n+1};{\bm{\phi}}),
$$

where $\Phi(\cdots;{\bm{\phi}})$ denotes the update function of a one-step ODE solver for the empirical PF ODE defined by the score model ${\bm{s}}_{\bm{\phi}}({\mathbf{x}},t)$. By default, $\mathbb{E}[\cdot]$ denotes the expectation over all relevant random variables in the expression.

<span id="section-8-2"></span>

### 8.2 Consistency Distillation

<span id="theorem-01-appendix"></span>

**Theorem 1.** Let $\Delta t\coloneqq\max_{n\in\llbracket 1,N-1\rrbracket}\{|t_{n+1}-t_{n}|\}$, and ${\bm{f}}(\cdot,\cdot;{\bm{\phi}})$ be the consistency function of the empirical PF ODE in [Equation 3](#equation-03). Assume ${\bm{f}}_{\bm{\theta}}$ satisfies the Lipschitz condition: there exists $L>0$ such that for all $t\in[\epsilon,T]$, ${\mathbf{x}}$, and ${\mathbf{y}}$, we have $\|{\bm{f}}_{\bm{\theta}}({\mathbf{x}},t)-{\bm{f}}_{\bm{\theta}}({\mathbf{y}},t)\|_{2}\leq L\|{\mathbf{x}}-{\mathbf{y}}\|_{2}$. Assume further that for all $n\in\llbracket 1,N-1\rrbracket$, the ODE solver called at $t_{n+1}$ has local error uniformly bounded by $O((t_{n+1}-t_{n})^{p+1})$ with $p\geq 1$. Then, if $\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}};{\bm{\phi}})=0$, we have

$$
\sup_{n,{\mathbf{x}}}\|{\bm{f}}_{{\bm{\theta}}}({\mathbf{x}},t_{n})-{\bm{f}}({\mathbf{x}},t_{n};{\bm{\phi}})\|_{2}=O((\Delta t)^{p}).
$$

::: details Proof
From $\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}};{\bm{\phi}})=0$, we have

<span id="equation-11"></span>

$$
\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}};{\bm{\phi}})=\mathbb{E}[\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({{\mathbf{x}}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))]=0.
$$

According to the definition, we have $p_{t_{n}}({\mathbf{x}}_{t_{n}})=p_{\text{data}}({\mathbf{x}})\otimes\mathcal{N}(\bm{0},t_{n}^{2}{\bm{I}})$ where $t_{n}\geq\epsilon>0$. It follows that $p_{t_{n}}({\mathbf{x}}_{t_{n}})>0$ for every ${\mathbf{x}}_{t_{n}}$ and $1\leq n\leq N$. Therefore, [Equation 11](#equation-11) entails

<span id="equation-12"></span>

$$
\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({{\mathbf{x}}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))\equiv 0.
$$

Because $\lambda(\cdot)>0$ and $d({\mathbf{x}},{\mathbf{y}})=0\Leftrightarrow{\mathbf{x}}={\mathbf{y}}$, this further implies that

<span id="equation-13"></span>

$$
{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})\equiv{\bm{f}}_{{\bm{\theta}}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}).
$$

Now let ${\bm{e}}_{n}$ represent the error vector at $t_{n}$, which is defined as

$$
{\bm{e}}_{n}\coloneqq{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n}},t_{n})-{\bm{f}}({\mathbf{x}}_{t_{n}},t_{n};{\bm{\phi}}).
$$

We can easily derive the following recursion relation

<span id="equation-14"></span>

$$
\begin{aligned}
{\bm{e}}_{n+1} & ={\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})-{\bm{f}}({\mathbf{x}}_{t_{n+1}},t_{n+1};{\bm{\phi}}) \\
\mathrel{{\mathop{=}\limits}}{\bm{f}}_{\bm{\theta}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n})-{\bm{f}}({\mathbf{x}}_{t_{n}},t_{n};{\bm{\phi}}) \\
={\bm{f}}_{\bm{\theta}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n})-{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n}},t_{n})+{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n}},t_{n})-{\bm{f}}({\mathbf{x}}_{t_{n}},t_{n};{\bm{\phi}}) \\
={\bm{f}}_{\bm{\theta}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n})-{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n}},t_{n})+{\bm{e}}_{n},
\end{aligned}
$$

where (i) is due to [Equation 13](#equation-13) and ${\bm{f}}({\mathbf{x}}_{t_{n+1}},t_{n+1};{\bm{\phi}})={\bm{f}}({\mathbf{x}}_{t_{n}},t_{n};{\bm{\phi}})$. Because ${\bm{f}}_{\bm{\theta}}(\cdot,t_{n})$ has Lipschitz constant $L$, we have

$$
\begin{aligned}
\|{\bm{e}}_{n+1}\|_{2} & \leq\|{\bm{e}}_{n}\|_{2}+L\|\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}}-{\mathbf{x}}_{t_{n}}\|_{2} \\
\mathrel{{\mathop{=}\limits}}\|{\bm{e}}_{n}\|_{2}+L\cdot O((t_{n+1}-t_{n})^{p+1}) \\
=\|{\bm{e}}_{n}\|_{2}+O((t_{n+1}-t_{n})^{p+1}),
\end{aligned}
$$

where (i) holds because the ODE solver has local error bounded by $O((t_{n+1}-t_{n})^{p+1})$. In addition, we observe that ${\bm{e}}_{1}=\bm{0}$, because

$$
\begin{aligned}
{\bm{e}}_{1} & ={\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{1}},t_{1})-{\bm{f}}({\mathbf{x}}_{t_{1}},t_{1};{\bm{\phi}}) \\
\mathrel{{\mathop{=}\limits}}{\mathbf{x}}_{t_{1}}-{\bm{f}}({\mathbf{x}}_{t_{1}},t_{1};{\bm{\phi}}) \\
\mathrel{{\mathop{=}\limits}}{\mathbf{x}}_{t_{1}}-{\mathbf{x}}_{t_{1}} \\
=\bm{0}.
\end{aligned}
$$

Here (i) is true because the consistency model is parameterized such that ${\bm{f}}({\mathbf{x}}_{t_{1}},t_{1};{\bm{\phi}})={\mathbf{x}}_{t_{1}}$ and (ii) is entailed by the definition of ${\bm{f}}(\cdot,\cdot;{\bm{\phi}})$. This allows us to perform induction on the recursion formula [Equation 14](#equation-14) to obtain

$$
\begin{aligned}
\|{\bm{e}}_{n}\|_{2} & \leq\|{\bm{e}}_{1}\|_{2}+\sum_{k=1}^{n-1}O((t_{k+1}-t_{k})^{p+1}) \\
=\sum_{k=1}^{n-1}O((t_{k+1}-t_{k})^{p+1}) \\
=\sum_{k=1}^{n-1}(t_{k+1}-t_{k})O((t_{k+1}-t_{k})^{p}) \\
\leq\sum_{k=1}^{n-1}(t_{k+1}-t_{k})O((\Delta t)^{p}) \\
=O((\Delta t)^{p})\sum_{k=1}^{n-1}(t_{k+1}-t_{k}) \\
=O((\Delta t)^{p})(t_{n}-t_{1}) \\
\leq O((\Delta t)^{p})(T-\epsilon) \\
=O((\Delta t)^{p}),
\end{aligned}
$$

which completes the proof.
:::

<span id="section-8-3"></span>

### 8.3 Consistency Training

The following lemma provides an unbiased estimator for the score function, which is crucial to our proof for [Theorem 2](#theorem-02).

<span id="lemma-01"></span>

**Lemma 1.** Let ${\mathbf{x}}\sim p_{\text{data}}({\mathbf{x}})$, ${\mathbf{x}}_{t}\sim\mathcal{N}({\mathbf{x}};t^{2}{\bm{I}})$, and $p_{t}({\mathbf{x}}_{t})=p_{\text{data}}({\mathbf{x}})\otimes\mathcal{N}(\bm{0},t^{2}{\bm{I}})$. We have $\nabla\log p_{t}({\mathbf{x}})=-\mathbb{E}[\frac{{\mathbf{x}}_{t}-{\mathbf{x}}}{t^{2}}\mid{\mathbf{x}}_{t}]$.

::: details Proof
According to the definition of $p_{t}({\mathbf{x}}_{t})$, we have $\nabla\log p_{t}({\mathbf{x}}_{t})=\nabla_{{\mathbf{x}}_{t}}\log\int p_{\text{data}}({\mathbf{x}})p({\mathbf{x}}_{t}\mid{\mathbf{x}})\mathop{}\!\mathrm{d}{\mathbf{x}}$, where $p({\mathbf{x}}_{t}\mid{\mathbf{x}})=\mathcal{N}({\mathbf{x}}_{t};{\mathbf{x}},t^{2}{\bm{I}})$. This expression can be further simplified to yield

$$
\begin{aligned}
\nabla\log p_{t}({\mathbf{x}}_{t}) & =\frac{\int p_{\text{data}}({\mathbf{x}})\nabla_{{\mathbf{x}}_{t}}p({\mathbf{x}}_{t}\mid{\mathbf{x}})\mathop{}\!\mathrm{d}{\mathbf{x}}}{\int p_{\text{data}}({\mathbf{x}})p({\mathbf{x}}_{t}\mid{\mathbf{x}})\mathop{}\!\mathrm{d}{\mathbf{x}}} \\
=\frac{\int p_{\text{data}}({\mathbf{x}})p({\mathbf{x}}_{t}\mid{\mathbf{x}})\nabla_{{\mathbf{x}}_{t}}\log p({\mathbf{x}}_{t}\mid{\mathbf{x}})\mathop{}\!\mathrm{d}{\mathbf{x}}}{\int p_{\text{data}}({\mathbf{x}})p({\mathbf{x}}_{t}\mid{\mathbf{x}})\mathop{}\!\mathrm{d}{\mathbf{x}}} \\
=\frac{\int p_{\text{data}}({\mathbf{x}})p({\mathbf{x}}_{t}\mid{\mathbf{x}})\nabla_{{\mathbf{x}}_{t}}\log p({\mathbf{x}}_{t}\mid{\mathbf{x}})\mathop{}\!\mathrm{d}{\mathbf{x}}}{p_{t}({\mathbf{x}}_{t})} \\
=\int\frac{p_{\text{data}}({\mathbf{x}})p({\mathbf{x}}_{t}\mid{\mathbf{x}})}{p_{t}({\mathbf{x}}_{t})}\nabla_{{\mathbf{x}}_{t}}\log p({\mathbf{x}}_{t}\mid{\mathbf{x}})\mathop{}\!\mathrm{d}{\mathbf{x}} \\
\mathrel{{\mathop{=}\limits}}\int p({\mathbf{x}}\mid{\mathbf{x}}_{t})\nabla_{{\mathbf{x}}_{t}}\log p({\mathbf{x}}_{t}\mid{\mathbf{x}})\mathop{}\!\mathrm{d}{\mathbf{x}} \\
=\mathbb{E}[\nabla_{{\mathbf{x}}_{t}}\log p({\mathbf{x}}_{t}\mid{\mathbf{x}})\mid{\mathbf{x}}_{t}] \\
=-\mathbb{E}\left[\frac{{\mathbf{x}}_{t}-{\mathbf{x}}}{t^{2}}\mid{\mathbf{x}}_{t}\right],
\end{aligned}
$$

where (i) is due to Bayes’ rule.
:::

<span id="theorem-02-appendix"></span>

**Theorem 2.** Let $\Delta t\coloneqq\max_{n\in\llbracket 1,N-1\rrbracket}\{|t_{n+1}-t_{n}|\}$. Assume $d$ and ${\bm{f}}_{{\bm{\theta}}^{-}}$ are both twice continuously differentiable with bounded second derivatives, the weighting function $\lambda(\cdot)$ is bounded, and $\mathbb{E}[\|\nabla\log p_{t_{n}}({\mathbf{x}}_{t_{n}})\|_{2}^{2}]<\infty$. Assume further that we use the Euler ODE solver, and the pre-trained score model matches the ground truth, *i.e*., $\forall t\in[\epsilon,T]:{\bm{s}}_{{\bm{\phi}}}({\mathbf{x}},t)\equiv\nabla\log p_{t}({\mathbf{x}})$. Then,

$$
\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})=\mathcal{L}_{\text{CT}}^{N}({\bm{\theta}},{\bm{\theta}}^{-})+o(\Delta t),
$$

where the expectation is taken with respect to ${\mathbf{x}}\sim p_{\text{data}}$, $n\sim\mathcal{U}\llbracket 1,N-1\rrbracket$, and ${\mathbf{x}}_{t_{n+1}}\sim\mathcal{N}({\mathbf{x}};t_{n+1}^{2}{\bm{I}})$. The consistency training objective, denoted by $\mathcal{L}_{\text{CT}}^{N}({\bm{\theta}},{\bm{\theta}}^{-})$, is defined as

$$
\mathbb{E}[\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n}))],
$$

where ${\mathbf{z}}\sim\mathcal{N}(\bm{0},{\bm{I}})$. Moreover, $\mathcal{L}_{\text{CT}}^{N}({\bm{\theta}},{\bm{\theta}}^{-})\geq O(\Delta t)$ if $\inf_{N}\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})>0$.

::: details Proof
With Taylor expansion, we have

<span id="equation-15"></span>

$$
\begin{aligned}
\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})=\mathbb{E}[\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n})] \\
= & \mathbb{E}[\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}}+(t_{n+1}-t_{n})t_{n+1}\nabla\log p_{t_{n+1}}({\mathbf{x}}_{t_{n+1}}),t_{n}))] \\
= & \mathbb{E}[\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})+\partial_{1}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})(t_{n+1}-t_{n})t_{n+1}\nabla\log p_{t_{n+1}}({\mathbf{x}}_{t_{n+1}}) \\
\qquad+\partial_{2}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})(t_{n}-t_{n+1})+o(|t_{n+1}-t_{n}|))] \\
= & \mathbb{E}\{\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1}))+\lambda(t_{n})\partial_{2}d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1}))[ \\
\quad\partial_{1}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})(t_{n+1}-t_{n})t_{n+1}\nabla\log p_{t_{n+1}}({\mathbf{x}}_{t_{n+1}})+\partial_{2}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})(t_{n}-t_{n+1})+o(|t_{n+1}-t_{n}|)]\} \\
= & \mathbb{E}[\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1}))] \\
\quad+\mathbb{E}\{\lambda(t_{n})\partial_{2}d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1}))[\partial_{1}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})(t_{n+1}-t_{n})t_{n+1}\nabla\log p_{t_{n+1}}({\mathbf{x}}_{t_{n+1}})]\} \\
\qquad+\mathbb{E}\{\lambda(t_{n})\partial_{2}d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1}))[\partial_{2}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})(t_{n}-t_{n+1})]\}+\mathbb{E}[o(|t_{n+1}-t_{n}|)].
\end{aligned}
$$

Then, we apply [Lemma 1](#lemma-01) to [Equation 15](#equation-15) and use Taylor expansion in the reverse direction to obtain

<span id="equation-16"></span>

$$
\begin{aligned}
\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}}) \\
= & \mathbb{E}[\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1}))] \\
\quad+\mathbb{E}\left\{\lambda(t_{n})\partial_{2}d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1}))\left[\partial_{1}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})(t_{n}-t_{n+1})t_{n+1}\mathbb{E}\left[\frac{{\mathbf{x}}_{t_{n+1}}-{\mathbf{x}}}{t_{n+1}^{2}}\Big|{\mathbf{x}}_{t_{n+1}}\right]\right]\right\} \\
\qquad+\mathbb{E}\{\lambda(t_{n})\partial_{2}d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1}))[\partial_{2}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})(t_{n}-t_{n+1})]\}+\mathbb{E}[o(|t_{n+1}-t_{n}|)] \\
\mathrel{{\mathop{=}\limits}} & \mathbb{E}[\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1}))] \\
\quad+\mathbb{E}\left\{\lambda(t_{n})\partial_{2}d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1}))\left[\partial_{1}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})(t_{n}-t_{n+1})t_{n+1}\left(\frac{{\mathbf{x}}_{t_{n+1}}-{\mathbf{x}}}{t_{n+1}^{2}}\right)\right]\right\} \\
\qquad+\mathbb{E}\{\lambda(t_{n})\partial_{2}d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1}))[\partial_{2}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})(t_{n}-t_{n+1})]\}+\mathbb{E}[o(|t_{n+1}-t_{n}|)] \\
= & \mathbb{E}\bigg[\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})) \\
\quad+\lambda(t_{n})\partial_{2}d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1}))\left[\partial_{1}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})(t_{n}-t_{n+1})t_{n+1}\left(\frac{{\mathbf{x}}_{t_{n+1}}-{\mathbf{x}}}{t_{n+1}^{2}}\right)\right] \\
\quad+\lambda(t_{n})\partial_{2}d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1}))[\partial_{2}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})(t_{n}-t_{n+1})]+o(|t_{n+1}-t_{n}|)\bigg] \\
\qquad+\mathbb{E}[o(|t_{n+1}-t_{n}|)] \\
= & \mathbb{E}\left[\lambda(t_{n})d\left({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}\left({\mathbf{x}}_{t_{n+1}}+(t_{n}-t_{n+1})t_{n+1}\frac{{\mathbf{x}}_{t_{n+1}}-{\mathbf{x}}}{t_{n+1}^{2}},t_{n}\right)\right)\right]+\mathbb{E}[o(|t_{n+1}-t_{n}|)] \\
= & \mathbb{E}\left[\lambda(t_{n})d\left({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}\left({\mathbf{x}}_{t_{n+1}}+(t_{n}-t_{n+1})\frac{{\mathbf{x}}_{t_{n+1}}-{\mathbf{x}}}{t_{n+1}},t_{n}\right)\right)\right]+\mathbb{E}[o(|t_{n+1}-t_{n}|)] \\
= & \mathbb{E}\left[\lambda(t_{n})d\left({\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}\left({\mathbf{x}}+t_{n+1}{\mathbf{z}}+(t_{n}-t_{n+1}){\mathbf{z}},t_{n}\right)\right)\right]+\mathbb{E}[o(|t_{n+1}-t_{n}|)] \\
= & \mathbb{E}\left[\lambda(t_{n})d\left({\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}\left({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n}\right)\right)\right]+\mathbb{E}[o(|t_{n+1}-t_{n}|)] \\
= & \mathbb{E}\left[\lambda(t_{n})d\left({\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}\left({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n}\right)\right)\right]+\mathbb{E}[o(\Delta t)] \\
= & \mathbb{E}\left[\lambda(t_{n})d\left({\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}\left({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n}\right)\right)\right]+o(\Delta t) \\
= & \mathcal{L}_{\text{CT}}^{N}({\bm{\theta}},{\bm{\theta}}^{-})+o(\Delta t),
\end{aligned}
$$

where (i) is due to the law of total expectation, and ${\mathbf{z}}\coloneqq\frac{{\mathbf{x}}_{t_{n+1}}-{\mathbf{x}}}{t_{n+1}}\sim\mathcal{N}(\bm{0},{\bm{I}})$. This implies $\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})=\mathcal{L}_{\text{CT}}^{N}({\bm{\theta}},{\bm{\theta}}^{-})+o(\Delta t)$ and thus completes the proof for [Equation 9](#equation-09). Moreover, we have $\mathcal{L}_{\text{CT}}^{N}({\bm{\theta}},{\bm{\theta}}^{-})\geq O(\Delta t)$ whenever $\inf_{N}\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})>0$. Otherwise, $\mathcal{L}_{\text{CT}}^{N}({\bm{\theta}},{\bm{\theta}}^{-})<O(\Delta t)$ and thus $\lim_{\Delta t\to 0}\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})=0$, which is a clear contradiction to $\inf_{N}\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})>0$.
:::

<span id="remark-01"></span>

**Remark 1.** When the condition $\mathcal{L}_{\text{CT}}^{N}({\bm{\theta}},{\bm{\theta}}^{-})\geq O(\Delta t)$ is not satisfied, such as in the case where ${\bm{\theta}}^{-}=\operatorname{stopgrad}({\bm{\theta}})$, the validity of $\mathcal{L}_{\text{CT}}^{N}({\bm{\theta}},{\bm{\theta}}^{-})$ as a training objective for consistency models can still be justified by referencing the result provided in [Theorem 6](#theorem-06).

<span id="section-9"></span>

## 9 Continuous-Time Extensions

The consistency distillation and consistency training objectives can be generalized to hold for infinite time steps ($N\to\infty$) under suitable conditions.

<span id="section-9-1"></span>

### 9.1 Consistency Distillation in Continuous Time

Depending on whether ${\bm{\theta}}^{-}={\bm{\theta}}$ or ${\bm{\theta}}^{-}=\operatorname{stopgrad}({\bm{\theta}})$ (same as setting $\mu=0$), there are two possible continuous-time extensions for the consistency distillation objective $\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})$. Given a twice continuously differentiable metric function $d({\mathbf{x}},{\mathbf{y}})$, we define ${\bm{G}}({\mathbf{x}})$ as a matrix, whose $(i,j)$-th entry is given by

$$
[{\bm{G}}({\mathbf{x}})]_{ij}\coloneqq\frac{\partial^{2}d({\mathbf{x}},{\mathbf{y}})}{\partial y_{i}\partial y_{j}}\bigg|_{{\mathbf{y}}={\mathbf{x}}}.
$$

Similarly, we define ${\bm{H}}({\mathbf{x}})$ as

$$
[{\bm{H}}({\mathbf{x}})]_{ij}\coloneqq\frac{\partial^{2}d({\mathbf{y}},{\mathbf{x}})}{\partial y_{i}\partial y_{j}}\bigg|_{{\mathbf{y}}={\mathbf{x}}}.
$$

The matrices ${\bm{G}}$ and ${\bm{H}}$ play a crucial role in forming continuous-time objectives for consistency distillation. Additionally, we denote the Jacobian of ${\bm{f}}_{\bm{\theta}}({\mathbf{x}},t)$ with respect to ${\mathbf{x}}$ as $\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}},t)}{\partial{\mathbf{x}}}$.

When ${\bm{\theta}}^{-}={\bm{\theta}}$ (with no stopgrad operator), we have the following theoretical result.

<span id="theorem-03"></span>

**Theorem 3.** Let $t_{n}=\tau(\frac{n-1}{N-1})$, where $n\in\llbracket 1,N\rrbracket$, and $\tau(\cdot)$ is a strictly monotonic function with $\tau(0)=\epsilon$ and $\tau(1)=T$. Assume $\tau$ is continuously differentiable in $[0,1]$, $d$ is three times continuously differentiable with bounded third derivatives, and ${\bm{f}}_{{\bm{\theta}}}$ is twice continuously differentiable with bounded first and second derivatives. Assume further that the weighting function $\lambda(\cdot)$ is bounded, and $\sup_{{\mathbf{x}},t\in[\epsilon,T]}\|{\bm{s}}_{\bm{\phi}}({\mathbf{x}},t)\|_{2}<\infty$. Then with the Euler solver in consistency distillation, we have

<span id="equation-17"></span>

$$
\lim_{N\to\infty}(N-1)^{2}\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}};{\bm{\phi}})=\mathcal{L}_{\text{CD}}^{\infty}({\bm{\theta}},{\bm{\theta}};{\bm{\phi}}),
$$

where $\mathcal{L}_{\text{CD}}^{\infty}({\bm{\theta}},{\bm{\theta}};{\bm{\phi}})$ is defined as

<span id="equation-18"></span>

$$
\frac{1}{2}\mathbb{E}\left[\frac{\lambda(t)}{[(\tau^{-1})^{\prime}(t)]^{2}}\left(\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)}{\partial t}-t\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)}{\partial{\mathbf{x}}_{t}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t},t)\right)^{\top}{\bm{G}}({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t))\left(\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)}{\partial t}-t\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)}{\partial{\mathbf{x}}_{t}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t},t)\right)\right].
$$

Here the expectation above is taken over ${\mathbf{x}}\sim p_{\text{data}}$, $u\sim\mathcal{U}[0,1]$, $t=\tau(u)$, and ${\mathbf{x}}_{t}\sim\mathcal{N}({\mathbf{x}},t^{2}{\bm{I}})$.

::: details Proof
Let $\Delta u=\frac{1}{N-1}$ and $u_{n}=\frac{n-1}{N-1}$. First, we can derive the following equation with Taylor expansion:

<span id="equation-19"></span>

$$
\begin{aligned}
{\bm{f}}_{\bm{\theta}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n})-{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})={\bm{f}}_{{\bm{\theta}}}({\mathbf{x}}_{t_{n+1}}+t_{n+1}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t_{n+1}},t_{n+1})\tau^{\prime}(u_{n})\Delta u,t_{n})-{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}) \\
= & t_{n+1}\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial{\mathbf{x}}_{t_{n+1}}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t_{n+1}},t_{n+1})\tau^{\prime}(u_{n})\Delta u-\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial t_{n+1}}\tau^{\prime}(u_{n})\Delta u+O((\Delta u)^{2}),
\end{aligned}
$$

Note that $\tau^{\prime}(u_{n})=\frac{1}{\tau^{-1}(t_{n+1})}$. Then, we apply Taylor expansion to the consistency distillation loss, which gives

$$
\begin{aligned}
(N-1)^{2}\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}};{\bm{\phi}})=\frac{1}{(\Delta u)^{2}}\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}};{\bm{\phi}})=\frac{1}{(\Delta u)^{2}}\mathbb{E}[\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n})] \\
\mathrel{{\mathop{=}\limits}} & \begin{multlined}\frac{1}{2(\Delta u)^{2}}\bigg(\mathbb{E}\{\lambda(t_{n})\tau^{\prime}(u_{n})^{2}[{\bm{f}}_{\bm{\theta}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n})-{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})]^{\top}{\bm{G}}({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}))\\
\cdot[{\bm{f}}_{\bm{\theta}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n})-{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})]\}+\mathbb{E}[O(|\Delta u|^{3})]\bigg)\end{multlined} \\
\mathrel{{\mathop{=}\limits}} & \!\begin{multlined}\frac{1}{2}\mathbb{E}\bigg[\lambda(t_{n})\tau^{\prime}(u_{n})^{2}\left(\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial t_{n+1}}-t_{n+1}\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial{\mathbf{x}}_{t_{n+1}}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t_{n+1}},t_{n+1})\right)^{\top}{\bm{G}}({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}))\\
\cdot\bigg(\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial t_{n+1}}-t_{n+1}\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial{\mathbf{x}}_{t_{n+1}}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t_{n+1}},t_{n+1})\bigg)\bigg]+\mathbb{E}[O(|\Delta u|)]\end{multlined} \\
= & \!\begin{multlined}\frac{1}{2}\mathbb{E}\bigg[\frac{\lambda(t_{n})}{[(\tau^{-1})^{\prime}(t_{n})]^{2}}\left(\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial t_{n+1}}-t_{n+1}\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial{\mathbf{x}}_{t_{n+1}}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t_{n+1}},t_{n+1})\right)^{\top}{\bm{G}}({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}))\\
\cdot\bigg(\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial t_{n+1}}-t_{n+1}\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial{\mathbf{x}}_{t_{n+1}}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t_{n+1}},t_{n+1})\bigg)\bigg]+\mathbb{E}[O(|\Delta u|)]\end{multlined}
\end{aligned}
$$

where we obtain (i) by expanding $d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),\cdot)$ to second order and observing $d({\mathbf{x}},{\mathbf{x}})\equiv 0$ and $\nabla_{\mathbf{y}}d({\mathbf{x}},{\mathbf{y}})|_{{\mathbf{y}}={\mathbf{x}}}\equiv\bm{0}$. We obtain (ii) using [Equation 19](#equation-19). By taking the limit for both sides of Section B.1 as $\Delta u\to 0$ or equivalently $N\to\infty$, we arrive at [Equation 17](#equation-17), which completes the proof.
:::

<span id="remark-02"></span>

**Remark 2.** Although [Theorem 3](#theorem-03) assumes the Euler ODE solver for technical simplicity, we believe an analogous result can be derived for more general solvers, since all ODE solvers should perform similarly as $N\to\infty$. We leave a more general version of [Theorem 3](#theorem-03) as future work.

<span id="remark-03"></span>

**Remark 3.** [Theorem 3](#theorem-03) implies that consistency models can be trained by minimizing $\mathcal{L}_{\text{CD}}^{\infty}({\bm{\theta}},{\bm{\theta}};{\bm{\phi}})$. In particular, when $d({\mathbf{x}},{\mathbf{y}})=\|{\mathbf{x}}-{\mathbf{y}}\|_{2}^{2}$, we have

<span id="equation-26"></span>

$$
\mathcal{L}_{\text{CD}}^{\infty}({\bm{\theta}},{\bm{\theta}};{\bm{\phi}})=\mathbb{E}\left[\frac{\lambda(t)}{[(\tau^{-1})^{\prime}(t)]^{2}}\|\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)}{\partial t}-t\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)}{\partial{\mathbf{x}}_{t}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t},t)\|^{2}_{2}\right].
$$

However, this continuous-time objective requires computing Jacobian-vector products as a subroutine to evaluate the loss function, which can be slow and laborious to implement in deep learning frameworks that do not support forward-mode automatic differentiation.

<span id="remark-04"></span>

**Remark 4.** If ${\bm{f}}_{\bm{\theta}}({\mathbf{x}},t)$ matches the ground truth consistency function for the empirical PF ODE of ${\bm{s}}_{\bm{\phi}}({\mathbf{x}},t)$, then

$$
\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}},t)}{\partial t}-t\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}},t)}{\partial{\mathbf{x}}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}},t)\equiv 0
$$

and therefore $\mathcal{L}_{\text{CD}}^{\infty}({\bm{\theta}},{\bm{\theta}};{\bm{\phi}})=0$. This can be proved by noting that ${\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)\equiv{\mathbf{x}}_{\epsilon}$ for all $t\in[\epsilon,T]$, and then taking the time-derivative of this identity:

$$
\begin{aligned}
{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)\equiv{\mathbf{x}}_{\epsilon} \\
\Longleftrightarrow & \frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)}{\partial{\mathbf{x}}_{t}}\frac{\mathop{}\!\mathrm{d}{\mathbf{x}}_{t}}{\mathop{}\!\mathrm{d}t}+\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)}{\partial t}\equiv 0 \\
\Longleftrightarrow & \frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)}{\partial{\mathbf{x}}_{t}}[-t{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t},t)]+\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)}{\partial t}\equiv 0 \\
\Longleftrightarrow & \frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)}{\partial t}-t\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)}{\partial{\mathbf{x}}_{t}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t},t)\equiv 0.
\end{aligned}
$$

The above observation provides another motivation for $\mathcal{L}_{\text{CD}}^{\infty}({\bm{\theta}},{\bm{\theta}};{\bm{\phi}})$, as it is minimized if and only if the consistency model matches the ground truth consistency function.

For some metric functions, such as the $\ell_{1}$ norm, the Hessian ${\bm{G}}({\mathbf{x}})$ is zero so [Theorem 3](#theorem-03) is vacuous. Below we show that a non-vacuous statement holds for the $\ell_{1}$ norm with just a small modification of the proof for [Theorem 3](#theorem-03).

<span id="theorem-04"></span>

**Theorem 4.** Let $t_{n}=\tau(\frac{n-1}{N-1})$, where $n\in\llbracket 1,N\rrbracket$, and $\tau(\cdot)$ is a strictly monotonic function with $\tau(0)=\epsilon$ and $\tau(1)=T$. Assume $\tau$ is continuously differentiable in $[0,1]$, and ${\bm{f}}_{{\bm{\theta}}}$ is twice continuously differentiable with bounded first and second derivatives. Assume further that the weighting function $\lambda(\cdot)$ is bounded, and $\sup_{{\mathbf{x}},t\in[\epsilon,T]}\|{\bm{s}}_{\bm{\phi}}({\mathbf{x}},t)\|_{2}<\infty$. Suppose we use the Euler ODE solver, and set $d({\mathbf{x}},{\mathbf{y}})=\|{\mathbf{x}}-{\mathbf{y}}\|_{1}$ in consistency distillation. Then we have

<span id="equation-27"></span>

$$
\lim_{N\to\infty}(N-1)\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}};{\bm{\phi}})=\mathcal{L}_{\text{CD, $\ell_{1}$}}^{\infty}({\bm{\theta}},{\bm{\theta}};{\bm{\phi}}),
$$

where

$$
\mathcal{L}_{\text{CD, $\ell_{1}$}}^{\infty}({\bm{\theta}},{\bm{\theta}};{\bm{\phi}})\coloneqq\mathbb{E}\left[\frac{\lambda(t)}{(\tau^{-1})^{\prime}(t)}\| t\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)}{\partial{\mathbf{x}}_{t}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t},t)-\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)}{\partial t}\|_{1}\right]
$$

where the expectation above is taken over ${\mathbf{x}}\sim p_{\text{data}}$, $u\sim\mathcal{U}[0,1]$, $t=\tau(u)$, and ${\mathbf{x}}_{t}\sim\mathcal{N}({\mathbf{x}},t^{2}{\bm{I}})$.

::: details Proof
Let $\Delta u=\frac{1}{N-1}$ and $u_{n}=\frac{n-1}{N-1}$. We have

<span id="equation-28"></span>

$$
\begin{aligned}
(N-1)\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}};{\bm{\phi}})=\frac{1}{\Delta u}\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}};{\bm{\phi}})=\frac{1}{\Delta u}\mathbb{E}[\lambda(t_{n})\|{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n})\|_{1}] \\
\mathrel{{\mathop{=}\limits}} & \frac{1}{\Delta u}\mathbb{E}\left[\lambda(t_{n})\| t_{n+1}\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial{\mathbf{x}}_{t_{n+1}}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t_{n+1}},t_{n+1})\tau^{\prime}(u_{n})-\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial t_{n+1}}\tau^{\prime}(u_{n})+O((\Delta u)^{2})\|_{1}\right] \\
= & \mathbb{E}\left[\lambda(t_{n})\tau^{\prime}(u_{n})\| t_{n+1}\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial{\mathbf{x}}_{t_{n+1}}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t_{n+1}},t_{n+1})-\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial t_{n+1}}+O(\Delta u)\|_{1}\right] \\
= & \mathbb{E}\left[\frac{\lambda(t_{n})}{(\tau^{-1})^{\prime}(t_{n})}\| t_{n+1}\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial{\mathbf{x}}_{t_{n+1}}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t_{n+1}},t_{n+1})-\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial t_{n+1}}+O(\Delta u)\|_{1}\right]
\end{aligned}
$$

where (i) is obtained by plugging [Equation 19](#equation-19) into the previous equation. Taking the limit for both sides of [Equation 28](#equation-28) as $\Delta u\to 0$ or equivalently $N\to\infty$ leads to [Equation 27](#equation-27), which completes the proof.
:::

<span id="remark-05"></span>

**Remark 5.** According to [Theorem 4](#theorem-04), consistency models can be trained by minimizing $\mathcal{L}_{\text{CD, $\ell_{1}$}}^{\infty}({\bm{\theta}},{\bm{\theta}};{\bm{\phi}})$. Moreover, the same reasoning in [Remark 4](#remark-04) can be applied to show that $\mathcal{L}_{\text{CD, $\ell_{1}$}}^{\infty}({\bm{\theta}},{\bm{\theta}};{\bm{\phi}})=0$ if and only if ${\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)={\mathbf{x}}_{\epsilon}$ for all ${\mathbf{x}}_{t}\in\mathbb{R}^{d}$ and $t\in[\epsilon,T]$.

In the second case where ${\bm{\theta}}^{-}=\operatorname{stopgrad}({\bm{\theta}})$, we can derive a so-called “pseudo-objective” whose gradient matches the gradient of $\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})$ in the limit of $N\to\infty$. Minimizing this pseudo-objective with gradient descent gives another way to train consistency models via distillation. This pseudo-objective is provided by the theorem below.

<span id="theorem-05"></span>

**Theorem 5.** Let $t_{n}=\tau(\frac{n-1}{N-1})$, where $n\in\llbracket 1,N\rrbracket$, and $\tau(\cdot)$ is a strictly monotonic function with $\tau(0)=\epsilon$ and $\tau(1)=T$. Assume $\tau$ is continuously differentiable in $[0,1]$, $d$ is three times continuously differentiable with bounded third derivatives, and ${\bm{f}}_{{\bm{\theta}}}$ is twice continuously differentiable with bounded first and second derivatives. Assume further that the weighting function $\lambda(\cdot)$ is bounded, $\sup_{{\mathbf{x}},t\in[\epsilon,T]}\|{\bm{s}}_{\bm{\phi}}({\mathbf{x}},t)\|_{2}<\infty$, and $\sup_{{\mathbf{x}},t\in[\epsilon,T]}\|\nabla_{\bm{\theta}}{\bm{f}}_{\bm{\theta}}({\mathbf{x}},t)\|_{2}<\infty$. Suppose we use the Euler ODE solver, and ${\bm{\theta}}^{-}=\operatorname{stopgrad}({\bm{\theta}})$ in consistency distillation. Then,

<span id="equation-29"></span>

$$
\lim_{N\to\infty}(N-1)\nabla_{\bm{\theta}}\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})=\nabla_{\bm{\theta}}\mathcal{L}_{\text{CD}}^{\infty}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}}),
$$

where

<span id="equation-30"></span>

$$
\mathcal{L}_{\text{CD}}^{\infty}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})\coloneqq\mathbb{E}\left[\frac{\lambda(t)}{(\tau^{-1})^{\prime}(t)}{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t},t))\left(\frac{\partial{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t},t)}{\partial t}-t\frac{\partial{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t},t)}{\partial{\mathbf{x}}_{t}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t},t)\right)\right].
$$

Here the expectation above is taken over ${\mathbf{x}}\sim p_{\text{data}}$, $u\sim\mathcal{U}[0,1]$, $t=\tau(u)$, and ${\mathbf{x}}_{t}\sim\mathcal{N}({\mathbf{x}},t^{2}{\bm{I}})$.

::: details Proof
We denote $\Delta u=\frac{1}{N-1}$ and $u_{n}=\frac{n-1}{N-1}$. First, we leverage Taylor series expansion to obtain

<span id="equation-33"></span>

$$
\begin{aligned}
(N-1)\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})=\frac{1}{\Delta u}\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})=\frac{1}{\Delta u}\mathbb{E}[\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n})] \\
\mathrel{{\mathop{=}\limits}} & \begin{multlined}\frac{1}{2\Delta u}\bigg(\mathbb{E}\{\lambda(t_{n})[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n})]^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))\\
\cdot[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n})]\}+\mathbb{E}[O(|\Delta u|^{3})]\bigg)\end{multlined} \\
= & \frac{1}{2\Delta u}\mathbb{E}\{\lambda(t_{n})[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n})]^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n})]\}+\mathbb{E}[O(|\Delta u|^{2})]
\end{aligned}
$$

where (i) is derived by expanding $d(\cdot,{\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))$ to second order and leveraging $d({\mathbf{x}},{\mathbf{x}})\equiv 0$ and $\nabla_{\mathbf{y}}d({\mathbf{y}},{\mathbf{x}})|_{{\mathbf{y}}={\mathbf{x}}}\equiv\bm{0}$. Next, we compute the gradient of [Equation 33](#equation-33) with respect to ${\bm{\theta}}$ and simplify the result to obtain

$$
\begin{aligned}
(N-1)\nabla_{\bm{\theta}}\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})=\frac{1}{\Delta u}\nabla_{\bm{\theta}}\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}}) \\
= & \frac{1}{2\Delta u}\nabla_{\bm{\theta}}\mathbb{E}\{\lambda(t_{n})[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n})]^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n})]\}+\mathbb{E}[O(|\Delta u|^{2})] \\
\mathrel{{\mathop{=}\limits}} & \frac{1}{\Delta u}\mathbb{E}\{\lambda(t_{n})[\nabla_{\bm{\theta}}{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})]^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n})]\}+\mathbb{E}[O(|\Delta u|^{2})] \\
\mathrel{{\mathop{=}\limits}} & \!\begin{multlined}\frac{1}{\Delta u}\mathbb{E}\bigg\{\lambda(t_{n})[\nabla_{\bm{\theta}}{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})]^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))\bigg[t_{n+1}\frac{\partial{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial{\mathbf{x}}_{t_{n+1}}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t_{n+1}},t_{n+1})\tau^{\prime}(u_{n})\Delta u\\
-\frac{\partial{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial t_{n+1}}\tau^{\prime}(u_{n})\Delta u\bigg]\bigg\}+\mathbb{E}[O(|\Delta u|)]\end{multlined} \\
= & \!\begin{multlined}\mathbb{E}\bigg\{\lambda(t_{n})[\nabla_{\bm{\theta}}{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})]^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))\bigg[t_{n+1}\frac{\partial{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial{\mathbf{x}}_{t_{n+1}}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t_{n+1}},t_{n+1})\tau^{\prime}(u_{n})\\
-\frac{\partial{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial t_{n+1}}\tau^{\prime}(u_{n})\bigg]\bigg\}+\mathbb{E}[O(|\Delta u|)]\end{multlined} \\
= & \!\begin{multlined}\nabla_{\bm{\theta}}\mathbb{E}\bigg\{\lambda(t_{n})[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})]^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))\bigg[t_{n+1}\frac{\partial{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial{\mathbf{x}}_{t_{n+1}}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t_{n+1}},t_{n+1})\tau^{\prime}(u_{n})\\
-\frac{\partial{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial t_{n+1}}\tau^{\prime}(u_{n})\bigg]\bigg\}+\mathbb{E}[O(|\Delta u|)]\end{multlined} \\
= & \!\begin{multlined}\nabla_{\bm{\theta}}\mathbb{E}\bigg\{\frac{\lambda(t_{n})}{(\tau^{-1})^{\prime}(t_{n})}[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})]^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))\bigg[t_{n+1}\frac{\partial{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial{\mathbf{x}}_{t_{n+1}}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t_{n+1}},t_{n+1})\\
-\frac{\partial{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial t_{n+1}}\bigg]\bigg\}+\mathbb{E}[O(|\Delta u|)]\end{multlined}
\end{aligned}
$$

Here (i) results from the chain rule, and (ii) follows from [Equation 19](#equation-19) and ${\bm{f}}_{\bm{\theta}}({\mathbf{x}},t)\equiv{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}},t)$, since ${\bm{\theta}}^{-}=\operatorname{stopgrad}({\bm{\theta}})$. Taking the limit for both sides of Section B.1 as $\Delta u\to 0$ (or $N\to\infty$) yields [Equation 29](#equation-29), which completes the proof.
:::

<span id="remark-06"></span>

**Remark 6.** When $d({\mathbf{x}},{\mathbf{y}})=\|{\mathbf{x}}-{\mathbf{y}}\|_{2}^{2}$, the pseudo-objective $\mathcal{L}_{\text{CD}}^{\infty}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})$ can be simplified to

<span id="equation-42"></span>

$$
\mathcal{L}_{\text{CD}}^{\infty}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})=2\mathbb{E}\left[\frac{\lambda(t)}{(\tau^{-1})^{\prime}(t)}{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)^{\top}\left(\frac{\partial{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t},t)}{\partial t}-t\frac{\partial{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t},t)}{\partial{\mathbf{x}}_{t}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t},t)\right)\right].
$$

<span id="remark-07"></span>

**Remark 7.** The objective $\mathcal{L}_{\text{CD}}^{\infty}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})$ defined in [Theorem 5](#theorem-05) is only meaningful in terms of its gradient—one cannot measure the progress of training by tracking the value of $\mathcal{L}_{\text{CD}}^{\infty}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})$, but can still apply gradient descent to this objective to distill consistency models from pre-trained diffusion models. Because this objective is not a typical loss function, we refer to it as the “pseudo-objective” for consistency distillation.

<span id="remark-08"></span>

**Remark 8.** Following the same reasoning in [Remark 4](#remark-04), we can easily derive that $\mathcal{L}_{\text{CD}}^{\infty}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})=0$ and $\nabla_{\bm{\theta}}\mathcal{L}_{\text{CD}}^{\infty}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})=\bm{0}$ if ${\bm{f}}_{\bm{\theta}}({\mathbf{x}},t)$ matches the ground truth consistency function for the empirical PF ODE that involves ${\bm{s}}_{\bm{\phi}}({\mathbf{x}},t)$. However, the converse does not hold true in general. This distinguishes $\mathcal{L}_{\text{CD}}^{\infty}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})$ from $\mathcal{L}_{\text{CD}}^{\infty}({\bm{\theta}},{\bm{\theta}};{\bm{\phi}})$, the latter of which is a true loss function.

<span id="section-9-2"></span>

### 9.2 Consistency Training in Continuous Time

A remarkable observation is that the pseudo-objective in [Theorem 5](#theorem-05) can be estimated without any pre-trained diffusion models, which enables direct consistency training of consistency models. More precisely, we have the following result.

<span id="theorem-06"></span>

**Theorem 6.** Let $t_{n}=\tau(\frac{n-1}{N-1})$, where $n\in\llbracket 1,N\rrbracket$, and $\tau(\cdot)$ is a strictly monotonic function with $\tau(0)=\epsilon$ and $\tau(1)=T$. Assume $\tau$ is continuously differentiable in $[0,1]$, $d$ is three times continuously differentiable with bounded third derivatives, and ${\bm{f}}_{{\bm{\theta}}}$ is twice continuously differentiable with bounded first and second derivatives. Assume further that the weighting function $\lambda(\cdot)$ is bounded, $\mathbb{E}[\|\nabla\log p_{t_{n}}({\mathbf{x}}_{t_{n}})\|_{2}^{2}]<\infty$, $\sup_{{\mathbf{x}},t\in[\epsilon,T]}\|\nabla_{\bm{\theta}}{\bm{f}}_{\bm{\theta}}({\mathbf{x}},t)\|_{2}<\infty$, and ${\bm{\phi}}$ represents diffusion model parameters that satisfy ${\bm{s}}_{\bm{\phi}}({\mathbf{x}},t)\equiv\nabla\log p_{t}({\mathbf{x}})$. Then if ${\bm{\theta}}^{-}=\operatorname{stopgrad}({\bm{\theta}})$, we have

<span id="equation-43"></span>

$$
\lim_{N\to\infty}(N-1)\nabla_{\bm{\theta}}\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})=\lim_{N\to\infty}(N-1)\nabla_{\bm{\theta}}\mathcal{L}_{\text{CT}}^{N}({\bm{\theta}},{\bm{\theta}}^{-})=\nabla_{\bm{\theta}}\mathcal{L}_{\text{CT}}^{\infty}({\bm{\theta}},{\bm{\theta}}^{-}),
$$

where $\mathcal{L}^{N}_{\text{CD}}$ uses the Euler ODE solver, and

<span id="equation-44"></span>

$$
\mathcal{L}_{\text{CT}}^{\infty}({\bm{\theta}},{\bm{\theta}}^{-})\coloneqq\mathbb{E}\left[\frac{\lambda(t)}{(\tau^{-1})^{\prime}(t)}{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t},t))\left(\frac{\partial{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t},t)}{\partial t}+\frac{\partial{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t},t)}{\partial{\mathbf{x}}_{t}}\cdot\frac{{\mathbf{x}}_{t}-{\mathbf{x}}}{t}\right)\right].
$$

Here the expectation above is taken over ${\mathbf{x}}\sim p_{\text{data}}$, $u\sim\mathcal{U}[0,1]$, $t=\tau(u)$, and ${\mathbf{x}}_{t}\sim\mathcal{N}({\mathbf{x}},t^{2}{\bm{I}})$.

::: details Proof
The proof mostly follows that of [Theorem 5](#theorem-05). First, we leverage Taylor series expansion to obtain

$$
\begin{aligned}
(N-1)\mathcal{L}_{\text{CT}}^{N}({\bm{\theta}},{\bm{\theta}}^{-})=\frac{1}{\Delta u}\mathcal{L}_{\text{CT}}^{N}({\bm{\theta}},{\bm{\theta}}^{-})=\frac{1}{\Delta u}\mathbb{E}[\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n}))] \\
\mathrel{{\mathop{=}\limits}} & \begin{multlined}\frac{1}{2\Delta u}\bigg(\mathbb{E}\{\lambda(t_{n})[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n})]^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n}))\\
\cdot[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n})]\}+\mathbb{E}[O(|\Delta u|^{3})]\bigg)\end{multlined} \\
= & \begin{multlined}\frac{1}{2\Delta u}\mathbb{E}\{\lambda(t_{n})[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n})]^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n}))\\
\cdot[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n})]\}+\mathbb{E}[O(|\Delta u|^{2})]\end{multlined}
\end{aligned}
$$

where ${\mathbf{z}}\sim\mathcal{N}(\bm{0},{\bm{I}})$, (i) is derived by first expanding $d(\cdot,{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n}))$ to second order, and then noting that $d({\mathbf{x}},{\mathbf{x}})\equiv 0$ and $\nabla_{\mathbf{y}}d({\mathbf{y}},{\mathbf{x}})|_{{\mathbf{y}}={\mathbf{x}}}\equiv\bm{0}$. Next, we compute the gradient of Section B.2 with respect to ${\bm{\theta}}$ and simplify the result to obtain

<span id="equation-59"></span>

$$
\begin{aligned}
(N-1)\nabla_{\bm{\theta}}\mathcal{L}_{\text{CT}}^{N}({\bm{\theta}},{\bm{\theta}}^{-})=\frac{1}{\Delta u}\nabla_{\bm{\theta}}\mathcal{L}_{\text{CT}}^{N}({\bm{\theta}},{\bm{\theta}}^{-}) \\
= & \begin{multlined}\frac{1}{2\Delta u}\nabla_{\bm{\theta}}\mathbb{E}\{\lambda(t_{n})[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n})]^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n}))\\
\cdot[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n})]\}+\mathbb{E}[O(|\Delta u|^{2})]\end{multlined} \\
\mathrel{{\mathop{=}\limits}} & \begin{multlined}\frac{1}{\Delta u}\mathbb{E}\{\lambda(t_{n})[\nabla_{\bm{\theta}}{\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1})]^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n}))\\
\cdot[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n})]\}+\mathbb{E}[O(|\Delta u|^{2})]\end{multlined} \\
\mathrel{{\mathop{=}\limits}} & \begin{multlined}\frac{1}{\Delta u}\mathbb{E}\bigg\{\lambda(t_{n})[\nabla_{\bm{\theta}}{\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1})]^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n}))\bigg[\tau^{\prime}(u_{n})\Delta u\partial_{1}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n}){\mathbf{z}}\\
+\partial_{2}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n})\tau^{\prime}(u_{n})\Delta u\bigg]\bigg\}+\mathbb{E}[O(|\Delta u|)]\end{multlined} \\
= & \begin{multlined}\mathbb{E}\bigg\{\lambda(t_{n})\tau^{\prime}(u_{n})[\nabla_{\bm{\theta}}{\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1})]^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n}))\bigg[\partial_{1}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n}){\mathbf{z}}\\
+\partial_{2}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n})\bigg]\bigg\}+\mathbb{E}[O(|\Delta u|)]\end{multlined} \\
= & \begin{multlined}\nabla_{\bm{\theta}}\mathbb{E}\bigg\{\lambda(t_{n})\tau^{\prime}(u_{n})[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1})]^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n}))\bigg[\partial_{1}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n}){\mathbf{z}}\\
+\partial_{2}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n})\bigg]\bigg\}+\mathbb{E}[O(|\Delta u|)]\end{multlined} \\
= & \nabla_{\bm{\theta}}\mathbb{E}\bigg\{\lambda(t_{n})\tau^{\prime}(u_{n})[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})]^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n}},t_{n}))\bigg[\partial_{1}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n}},t_{n})\frac{{\mathbf{x}}_{t_{n}}-{\mathbf{x}}}{t_{n}}+\partial_{2}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n}},t_{n})\bigg]\bigg\}+\mathbb{E}[O(|\Delta u|)] \\
= & \nabla_{\bm{\theta}}\mathbb{E}\bigg\{\frac{\lambda(t_{n})}{(\tau^{-1})^{\prime}(t_{n})}[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})]^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n}},t_{n}))\bigg[\partial_{1}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n}},t_{n})\frac{{\mathbf{x}}_{t_{n}}-{\mathbf{x}}}{t_{n}}+\partial_{2}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n}},t_{n})\bigg]\bigg\}+\mathbb{E}[O(|\Delta u|)]
\end{aligned}
$$

Here (i) results from the chain rule, and (ii) follows from Taylor expansion. Taking the limit for both sides of [Equation 59](#equation-59) as $\Delta u\to 0$ or $N\to\infty$ yields the second equality in [Equation 43](#equation-43).

Now we prove the first equality. Applying Taylor expansion again, we obtain

$$
\begin{aligned}
(N-1)\nabla_{\bm{\theta}}\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})=\frac{1}{\Delta u}\nabla_{\bm{\theta}}\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})=\frac{1}{\Delta u}\nabla_{\bm{\theta}}\mathbb{E}[\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))] \\
= & \frac{1}{\Delta u}\mathbb{E}[\lambda(t_{n})\nabla_{\bm{\theta}}d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))] \\
= & \frac{1}{\Delta u}\mathbb{E}[\lambda(t_{n})\nabla_{\bm{\theta}}{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})^{\top}\partial_{1}d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))] \\
= & \frac{1}{\Delta u}\begin{multlined}\mathbb{E}\bigg\{\lambda(t_{n})\nabla_{\bm{\theta}}{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})^{\top}\bigg[\partial_{1}d({\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}),{\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))\\
+{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))+O(|\Delta u|^{2})\bigg]\bigg\}\end{multlined} \\
= & \frac{1}{\Delta u}\mathbb{E}\{\lambda(t_{n})\nabla_{\bm{\theta}}{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})^{\top}[{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))]+O(|\Delta u|^{2})\} \\
= & \frac{1}{\Delta u}\mathbb{E}\{\lambda(t_{n})\nabla_{\bm{\theta}}{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})^{\top}[{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))({\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))]+O(|\Delta u|^{2})\} \\
\mathrel{{\mathop{=}\limits}} & \begin{multlined}\frac{1}{\Delta u}\mathbb{E}\{\lambda(t_{n})[\nabla_{\bm{\theta}}{\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1})]^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n}))\\
\cdot[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n})]\}+\mathbb{E}[O(|\Delta u|^{2})]\end{multlined}
\end{aligned}
$$

where (i) holds because ${\mathbf{x}}_{t_{n+1}}={\mathbf{x}}+t_{n+1}{\mathbf{z}}$ and $\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}}={\mathbf{x}}_{t_{n+1}}-(t_{n}-t_{n+1})t_{n+1}\frac{-({\mathbf{x}}_{t_{n+1}}-{\mathbf{x}})}{t_{n+1}^{2}}={\mathbf{x}}_{t_{n+1}}+(t_{n}-t_{n+1}){\mathbf{z}}={\mathbf{x}}+t_{n}{\mathbf{z}}$. Because (i) matches Section B.2, we can use the same reasoning procedure from Section B.2 to [Equation 59](#equation-59) to conclude $\lim_{N\to\infty}(N-1)\nabla_{\bm{\theta}}\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})=\lim_{N\to\infty}(N-1)\nabla_{\bm{\theta}}\mathcal{L}_{\text{CT}}^{N}({\bm{\theta}},{\bm{\theta}}^{-})$, completing the proof.
:::

<span id="remark-09"></span>

**Remark 9.** Note that $\mathcal{L}_{\text{CT}}^{\infty}({\bm{\theta}},{\bm{\theta}}^{-})$ does not depend on the diffusion model parameter ${\bm{\phi}}$ and hence can be optimized without any pre-trained diffusion models.

<span id="remark-10"></span>

**Remark 10.** When $d({\mathbf{x}},{\mathbf{y}})=\|{\mathbf{x}}-{\mathbf{y}}\|_{2}^{2}$, the continuous-time consistency training objective becomes

<span id="equation-60"></span>

$$
\mathcal{L}_{\text{CT}}^{\infty}({\bm{\theta}},{\bm{\theta}}^{-})=2\mathbb{E}\left[\frac{\lambda(t)}{(\tau^{-1})^{\prime}(t)}{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)^{\top}\left(\frac{\partial{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t},t)}{\partial t}+\frac{\partial{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t},t)}{\partial{\mathbf{x}}_{t}}\cdot\frac{{\mathbf{x}}_{t}-{\mathbf{x}}}{t}\right)\right].
$$

<span id="remark-11"></span>

**Remark 11.** Similar to $\mathcal{L}_{\text{CD}}^{\infty}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})$ in [Theorem 5](#theorem-05), $\mathcal{L}_{\text{CT}}^{\infty}({\bm{\theta}},{\bm{\theta}}^{-})$ is a pseudo-objective; one cannot track training by monitoring the value of $\mathcal{L}_{\text{CT}}^{\infty}({\bm{\theta}},{\bm{\theta}}^{-})$, but can still apply gradient descent on this loss function to train a consistency model ${\bm{f}}_{\bm{\theta}}({\mathbf{x}},t)$ directly from data. Moreover, the same observation in [Remark 8](#remark-08) holds true: $\mathcal{L}_{\text{CT}}^{\infty}({\bm{\theta}},{\bm{\theta}}^{-})=0$ and $\nabla_{\bm{\theta}}\mathcal{L}_{\text{CT}}^{\infty}({\bm{\theta}},{\bm{\theta}}^{-})=\bm{0}$ if ${\bm{f}}_{\bm{\theta}}({\mathbf{x}},t)$ matches the ground truth consistency function for the PF ODE.

<span id="section-9-3"></span>

### 9.3 Experimental Verifications

<span id="figure-07"></span>

![Figure 7. Comparing discrete consistency distillation/training algorithms with continuous counterparts.](../../papers/consistency-models/figure-07.png)

**Figure 7.** Comparing discrete consistency distillation/training algorithms with continuous counterparts.

To experimentally verify the efficacy of our continuous-time CD and CT objectives, we train consistency models with a variety of loss functions on CIFAR-10. All results are provided in [Figure 7](#figure-07). We set $\lambda(t)=(\tau^{-1})^{\prime}(t)$ for all continuous-time experiments. Other hyperparameters are the same as in [Table 3](#table-03). We occasionally modify some hyperparameters for improved performance. For distillation, we compare the following objectives:

- CD $(\ell_{2})$: Consistency distillation $\mathcal{L}^{N}_{\text{CD}}$ with $N=18$ and the $\ell_{2}$ metric.
- CD $(\ell_{1})$: Consistency distillation $\mathcal{L}^{N}_{\text{CD}}$ with $N=18$ and the $\ell_{1}$ metric. We set the learning rate to 2e-4.
- CD (LPIPS): Consistency distillation $\mathcal{L}^{N}_{\text{CD}}$ with $N=18$ and the LPIPS metric.
- CD<sup>∞</sup> $(\ell_{2})$: Consistency distillation $\mathcal{L}^{\infty}_{\text{CD}}$ in [Theorem 3](#theorem-03) with the $\ell_{2}$ metric. We set the learning rate to 1e-3 and dropout to 0.13.
- CD<sup>∞</sup> $(\ell_{1})$: Consistency distillation $\mathcal{L}^{\infty}_{\text{CD}}$ in [Theorem 4](#theorem-04) with the $\ell_{1}$ metric. We set the learning rate to 1e-3 and dropout to 0.3.
- CD<sup>∞</sup> (stopgrad, $\ell_{2}$): Consistency distillation $\mathcal{L}^{\infty}_{\text{CD}}$ in [Theorem 5](#theorem-05) with the $\ell_{2}$ metric. We set the learning rate to 5e-6.
- CD<sup>∞</sup> (stopgrad, LPIPS): Consistency distillation $\mathcal{L}^{\infty}_{\text{CD}}$ in [Theorem 5](#theorem-05) with the LPIPS metric. We set the learning rate to 5e-6.

We did not investigate using the LPIPS metric in [Theorem 3](#theorem-03) because minimizing the resulting objective would require back-propagating through second order derivatives of the VGG network used in LPIPS, which is computationally expensive and prone to numerical instability. As revealed by [Figure 7(a)](#figure-07), the stopgrad version of continuous-time distillation ([Theorem 5](#theorem-05)) works better than the non-stopgrad version ([Theorem 3](#theorem-03)) for both the LPIPS and $\ell_{2}$ metrics, and the LPIPS metric works the best for all distillation approaches. Additionally, discrete-time consistency distillation outperforms continuous-time consistency distillation, possibly due to the larger variance in continuous-time objectives, and the fact that one can use effective higher-order ODE solvers in discrete-time objectives.

For consistency training (CT), we find it important to initialize consistency models from a pre-trained EDM model in order to stabilize training when using continuous-time objectives. We hypothesize that this is caused by the large variance in our continuous-time loss functions. For fair comparison, we thus initialize all consistency models from the same pre-trained EDM model on CIFAR-10 for both discrete-time and continuous-time CT, even though the former works well with random initialization. We leave variance reduction techniques for continuous-time CT to future research.

We empirically compare the following objectives:

- CT (LPIPS): Consistency training $\mathcal{L}_{\text{CT}}^{N}$ with $N=120$ and the LPIPS metric. We set the learning rate to 4e-4, and the EMA decay rate for the target network to 0.99. We do not use the schedule functions for $N$ and $\mu$ here because they cause slower learning when the consistency model is initialized from a pre-trained EDM model.
- CT<sup>∞</sup> $(\ell_{2})$: Consistency training $\mathcal{L}^{\infty}_{\text{CT}}$ with the $\ell_{2}$ metric. We set the learning rate to 5e-6.
- CT<sup>∞</sup> (LPIPS): Consistency training $\mathcal{L}^{\infty}_{\text{CT}}$ with the LPIPS metric. We set the learning rate to 5e-6.

As shown in [Figure 7(b)](#figure-07), the LPIPS metric leads to improved performance for continuous-time CT. We also find that continuous-time CT outperforms discrete-time CT with the same LPIPS metric. This is likely due to the bias in discrete-time CT, as $\Delta t>0$ in [Theorem 2](#theorem-02) for discrete-time objectives, whereas continuous-time CT has no bias since it implicitly drives $\Delta t$ to $0$.

<span id="section-10"></span>

## 10 Additional Experimental Details

<span id="table-03"></span>

![Table 3. Hyperparameters used for training CD and CT models](../../papers/consistency-models/table-03.png)

**Table 3.** Hyperparameters used for training CD and CT models

**Model Architectures.** We follow [Son21, Dha21] for model architectures. Specifically, we use the NCSN++ architecture in [Son21] for all CIFAR-10 experiments, and take the corresponding network architectures from [Dha21] when performing experiments on ImageNet $64\times 64$, LSUN Bedroom $256\times 256$ and LSUN Cat $256\times 256$.

**Parameterization for Consistency Models.** We use the same architectures for consistency models as those used for EDMs. The only difference is we slightly modify the skip connections in EDM to ensure the boundary condition holds for consistency models. Recall that in [Section 3](#section-3) we propose to parameterize a consistency model in the following form:

$$
{\bm{f}}_{\bm{\theta}}({\mathbf{x}},t)=c_{\text{skip}}(t){\mathbf{x}}+c_{\text{out}}(t)F_{\bm{\theta}}({\mathbf{x}},t).
$$

In EDM [Kar22], authors choose

$$
c_{\text{skip}}(t)=\frac{\sigma_{\text{data}}^{2}}{t^{2}+\sigma_{\text{data}}^{2}},\quad c_{\text{out}}(t)=\frac{\sigma_{\text{data}}t}{\sqrt{\sigma_{\text{data}}^{2}+t^{2}}},
$$

where $\sigma_{\text{data}}=0.5$. However, this choice of $c_{\text{skip}}$ and $c_{\text{out}}$ does not satisfy the boundary condition when the smallest time instant $\epsilon\neq 0$. To remedy this issue, we modify them to

$$
c_{\text{skip}}(t)=\frac{\sigma_{\text{data}}^{2}}{(t-\epsilon)^{2}+\sigma_{\text{data}}^{2}},\quad c_{\text{out}}(t)=\frac{\sigma_{\text{data}}(t-\epsilon)}{\sqrt{\sigma_{\text{data}}^{2}+t^{2}}},
$$

which clearly satisfies $c_{\text{skip}}(\epsilon)=1$ and $c_{\text{out}}(\epsilon)=0$.

**Schedule Functions for Consistency Training.** As discussed in [Section 5](#section-5), consistency generation requires specifying schedule functions $N(\cdot)$ and $\mu(\cdot)$ for best performance. Throughout our experiments, we use schedule functions that take the form below:

$$
\begin{aligned}
N(k) & =\left\lceil\sqrt{\frac{k}{K}((s_{1}+1)^{2}-s_{0}^{2})+s_{0}^{2}}-1\right\rceil+1 \\
\mu(k) & =\exp\left(\frac{s_{0}\log\mu_{0}}{N(k)}\right),
\end{aligned}
$$

where $K$ denotes the total number of training iterations, $s_{0}$ denotes the initial discretization steps, $s_{1}>s_{0}$ denotes the target discretization steps at the end of training, and $\mu_{0}>0$ denotes the EMA decay rate at the beginning of model training.

**Training Details.** In both consistency distillation and progressive distillation, we distill EDMs [Kar22]. We trained these EDMs ourselves according to the specifications given in [Kar22]. The original EDM paper did not provide hyperparameters for the LSUN Bedroom $256\times 256$ and Cat $256\times 256$ datasets, so we mostly used the same hyperparameters as those for the ImageNet $64\times 64$ dataset. The difference is that we trained for 600k and 300k iterations for the LSUN Bedroom and Cat datasets respectively, and reduced the batch size from 4096 to 2048.

We used the same EMA decay rate for LSUN $256\times 256$ datasets as for the ImageNet $64\times 64$ dataset. For progressive distillation, we used the same training settings as those described in [Sal22] for CIFAR-10 and ImageNet $64\times 64$. Although the original paper did not test on LSUN $256\times 256$ datasets, we used the same settings for ImageNet $64\times 64$ and found them to work well.

In all distillation experiments, we initialized the consistency model with pre-trained EDM weights. For consistency training, we initialized the model randomly, just as we did for training the EDMs. We trained all consistency models with the Rectified Adam optimizer [Liu19d], with no learning rate decay or warm-up, and no weight decay. We also applied EMA to the weights of the online consistency models in both consistency distillation and consistency training, as well as to the weights of the training online consistency models according to [Kar22]. For LSUN $256\times 256$ datasets, we chose the EMA decay rate to be the same as that for ImageNet $64\times 64$, except for consistency distillation on LSUN Bedroom $256\times 256$, where we found that using zero EMA worked better.

When using the LPIPS metric on CIFAR-10 and ImageNet $64\times 64$, we rescale images to resolution $224\times 224$ with bilinear upsampling before feeding them to the LPIPS network. For LSUN $256\times 256$, we evaluated LPIPS without rescaling inputs. In addition, we performed horizontal flips for data augmentation for all models and on all datasets. We trained all models on a cluster of Nvidia A100 GPUs. Additional hyperparameters for consistency training and distillation are listed in [Table 3](#table-03).

<span id="section-11"></span>

## 11 Additional Results on Zero-Shot Image Editing

<span id="algorithm-04"></span>

<div class="paper-algorithm">

**Algorithm 4: Zero-Shot Image Editing.**

- **Input:** Consistency model ${\bm{f}}_{\bm{\theta}}(\cdot,\cdot)$, sequence of time points $t_{1}>t_{2}>\cdots>t_{N}$, reference image ${\mathbf{y}}$, invertible linear transformation ${\bm{A}}$, and binary image mask $\bm{\Omega}$.
- ${\mathbf{y}}\gets{\bm{A}}^{-1}[({\bm{A}}{\mathbf{y}})\odot(1-\bm{\Omega})+\bm{0}\odot\bm{\Omega}]$.
- Sample ${\mathbf{x}}\sim\mathcal{N}({\mathbf{y}},t_{1}^{2}{\bm{I}})$.
- ${\mathbf{x}}\gets{\bm{f}}_{\bm{\theta}}({\mathbf{x}},t_{1})$.
- ${\mathbf{x}}\gets{\bm{A}}^{-1}[({\bm{A}}{\mathbf{y}})\odot(1-\bm{\Omega})+({\bm{A}}{\mathbf{x}})\odot\bm{\Omega}]$.
- **For** $n=2$ **to** $N$:
  - Sample ${\mathbf{x}}\sim\mathcal{N}({\mathbf{x}},(t_{n}^{2}-\epsilon^{2}){\bm{I}})$.
  - ${\mathbf{x}}\gets{\bm{f}}_{\bm{\theta}}({\mathbf{x}},t_{n})$.
  - ${\mathbf{x}}\gets{\bm{A}}^{-1}[({\bm{A}}{\mathbf{y}})\odot(1-\bm{\Omega})+({\bm{A}}{\mathbf{x}})\odot\bm{\Omega}]$.
- **Output:** ${\mathbf{x}}$.

</div>

With consistency models, we can perform a variety of zero-shot image editing tasks. As an example, we present additional results on colorization ([Figure 8](#figure-08)), super-resolution ([Figure 9](#figure-09)), inpainting ([Figure 10](#figure-10)), interpolation ([Figure 11](#figure-11)), denoising ([Figure 12](#figure-12)), and stroke-guided image generation (SDEdit, [Men22], [Figure 13](#figure-13)). The consistency model used here is trained via consistency distillation on the LSUN Bedroom $256\times 256$.

All these image editing tasks, except for image interpolation and denoising, can be performed via a small modification to the multistep sampling algorithm in [Algorithm 1](#algorithm-01). The resulting pseudocode is provided in [Algorithm 4](#algorithm-04). Here ${\mathbf{y}}$ is a reference image that guides sample generation, $\bm{\Omega}$ is a binary mask, $\odot$ computes element-wise products, and ${\bm{A}}$ is an invertible linear transformation that maps images into a latent space where the conditional information in ${\mathbf{y}}$ is infused into the iterative generation procedure by masking with $\bm{\Omega}$. Unless otherwise stated, we choose

$$
t_{i}=\left(T^{1/\rho}+\frac{i-1}{N-1}(\epsilon^{1/\rho}-T^{1/\rho})\right)^{\rho}
$$

in our experiments, where $N=40$ for LSUN Bedroom $256\times 256$.

Below we describe how to perform each task using [Algorithm 4](#algorithm-04).

**Inpainting.** When using [Algorithm 4](#algorithm-04) for inpainting, we let ${\mathbf{y}}$ be an image where missing pixels are masked out, $\bm{\Omega}$ be a binary mask where 1 indicates the missing pixels, and ${\bm{A}}$ be the identity transformation.

**Colorization.** The algorithm for image colorization is similar, as colorization becomes a special case of inpainting once we transform data into a decoupled space. Specifically, let ${\mathbf{y}}\in\mathbb{R}^{h\times w\times 3}$ be a gray-scale image that we aim to colorize, where all channels of ${\mathbf{y}}$ are assumed to be the same, *i.e*., ${\mathbf{y}}[:,:,0]={\mathbf{y}}[:,:,1]={\mathbf{y}}[:,:,2]$ in NumPy notation. In our experiments, each channel of this gray scale image is obtained from a colorful image by averaging the RGB channels with

$$
0.2989R+0.5870G+0.1140B.
$$

We define $\bm{\Omega}\in\{0,1\}^{h\times w\times 3}$ to be a binary mask such that

$$
\bm{\Omega}[i,j,k]=\begin{cases}1,&\quad\text{$k=1$ or $2$}\\
0,&\quad\text{$k=0$}\end{cases}.
$$

Let ${\bm{Q}}\in\mathbb{R}^{3\times 3}$ be an orthogonal matrix whose first column is proportional to the vector $(0.2989,0.5870,0.1140)$. This orthogonal matrix can be obtained easily via QR decomposition, and we use the following in our experiments

$$
{\bm{Q}}=\begin{pmatrix}0.4471&-0.8204&0.3563\\
0.8780&0.4785&0\\
0.1705&-0.3129&-0.9343\end{pmatrix}.
$$

We then define the linear transformation ${\bm{A}}:{\mathbf{x}}\in\mathbb{R}^{h\times w\times 3}\mapsto{\mathbf{y}}\in\mathbb{R}^{h\times w\times 3}$, where

$$
{\mathbf{y}}[i,j,k]=\sum_{l=0}^{2}{\mathbf{x}}[i,j,l]{\bm{Q}}[l,k].
$$

Because ${\bm{Q}}$ is orthogonal, the inversion ${\bm{A}}^{-1}:{\mathbf{y}}\in\mathbb{R}^{h\times w}\mapsto{\mathbf{x}}\in\mathbb{R}^{h\times w\times 3}$ is easy to compute, where

$$
{\mathbf{x}}[i,j,k]=\sum_{l=0}^{2}{\mathbf{y}}[i,j,l]{\bm{Q}}[k,l].
$$

With ${\bm{A}}$ and $\bm{\Omega}$ defined as above, we can now use [Algorithm 4](#algorithm-04) for image colorization.

**Super-resolution.** With a similar strategy, we employ [Algorithm 4](#algorithm-04) for image super-resolution. For simplicity, we assume that the down-sampled image is obtained by averaging non-overlapping patches of size $p\times p$. Suppose the shape of full resolution images is $h\times w\times 3$. Let ${\mathbf{y}}\in\mathbb{R}^{h\times w\times 3}$ denote a low-resolution image naively up-sampled to full resolution, where pixels in each non-overlapping patch share the same value. Additionally, let $\bm{\Omega}\in\{0,1\}^{h/p\times w/p\times p^{2}\times 3}$ be a binary mask such that

$$
\bm{\Omega}[i,j,k,l]=\begin{cases}1,&\quad k\geq 1\\
0,&\quad k=0\end{cases}.
$$

Similar to image colorization, super-resolution requires an orthogonal matrix ${\bm{Q}}\in\mathbb{R}^{p^{2}\times p^{2}}$ whose first column is $(\frac{1}{p},\frac{1}{p},\cdots,\frac{1}{p})$. This orthogonal matrix can be obtained with QR decomposition. To perform super-resolution, we define the linear transformation ${\bm{A}}:{\mathbf{x}}\in\mathbb{R}^{h\times w\times 3}\mapsto{\mathbf{y}}\in\mathbb{R}^{h/p\times w/p\times p^{2}\times 3}$, where

$$
{\mathbf{y}}[i,j,k,l]=\sum_{m=0}^{p^{2}-1}{\mathbf{x}}[i\times p+(m-m\bmod p)/p,j\times p+m\bmod p,l]{\bm{Q}}[m,k].
$$

The inverse transformation ${\bm{A}}^{-1}:{\mathbf{y}}\in\mathbb{R}^{h/p\times w/p\times p^{2}\times 3}\mapsto{\mathbf{x}}\in\mathbb{R}^{h\times w\times 3}$ is easy to derive, with

$$
{\mathbf{x}}[i,j,k,l]=\sum_{m=0}^{p^{2}-1}{\mathbf{y}}[i\times p+(m-m\bmod p)/p,j\times p+m\bmod p,l]{\bm{Q}}[k,m].
$$

Above definitions of ${\bm{A}}$ and $\bm{\Omega}$ allow us to use [Algorithm 4](#algorithm-04) for image super-resolution.

**Stroke-guided image generation.** We can also use [Algorithm 4](#algorithm-04) for stroke-guided image generation as introduced in SDEdit [Men22]. Specifically, we let ${\mathbf{y}}\in\mathbb{R}^{h\times w\times 3}$ be a stroke painting. We set ${\bm{A}}={\bm{I}}$, and define $\bm{\Omega}\in\mathbb{R}^{h\times w\times 3}$ as a matrix of ones. In our experiments, we set $t_{1}=5.38$ and $t_{2}=2.24$, with $N=2$.

**Denoising.** It is possible to denoise images perturbed with various scales of Gaussian noise using a single consistency model. Suppose the input image ${\mathbf{x}}$ is perturbed with $\mathcal{N}(\bm{0};\sigma^{2}{\bm{I}})$. As long as $\sigma\in[\epsilon,T]$, we can evaluate ${\bm{f}}_{\bm{\theta}}({\mathbf{x}},\sigma)$ to produce the denoised image.

**Interpolation.** We can interpolate between two images generated by consistency models. Suppose the first sample ${\mathbf{x}}_{1}$ is produced by noise vector ${\mathbf{z}}_{1}$, and the second sample ${\mathbf{x}}_{2}$ is produced by noise vector ${\mathbf{z}}_{2}$. In other words, ${\mathbf{x}}_{1}={\bm{f}}_{\bm{\theta}}({\mathbf{z}}_{1},T)$ and ${\mathbf{x}}_{2}={\bm{f}}_{\bm{\theta}}({\mathbf{z}}_{2},T)$. To interpolate between ${\mathbf{x}}_{1}$ and ${\mathbf{x}}_{2}$, we first use spherical linear interpolation to get

$$
{\mathbf{z}}=\frac{\sin[(1-\alpha)\psi]}{\sin(\psi)}{\mathbf{z}}_{1}+\frac{\sin(\alpha\psi)}{\sin(\psi)}{\mathbf{z}}_{2},
$$

where $\alpha\in[0,1]$ and $\psi=\arccos(\frac{{\mathbf{z}}_{1}^{\top}{\mathbf{z}}_{2}}{\|{\mathbf{z}}_{1}\|_{2}\|{\mathbf{z}}_{2}\|_{2}})$, then evaluate ${\bm{f}}_{\bm{\theta}}({\mathbf{z}},T)$ to produce the interpolated image.

<span id="figure-08"></span>

![Figure 8. Gray-scale images (left), colorized images by a consistency model (middle), and ground truth (right).](../../papers/consistency-models/figure-08.png)

**Figure 8.** Gray-scale images (left), colorized images by a consistency model (middle), and ground truth (right).

<span id="figure-09"></span>

![Figure 9. Downsampled images of resolution $32\times 32$ (left), full resolution ($256\times 256$) images generated by a consistency model (middle), and ground truth images of resolution $256\times 256$ (right).](../../papers/consistency-models/figure-09.png)

**Figure 9.** Downsampled images of resolution $32\times 32$ (left), full resolution ($256\times 256$) images generated by a consistency model (middle), and ground truth images of resolution $256\times 256$ (right).

<span id="figure-10"></span>

![Figure 10. Masked images (left), imputed images by a consistency model (middle), and ground truth (right).](../../papers/consistency-models/figure-10.png)

**Figure 10.** Masked images (left), imputed images by a consistency model (middle), and ground truth (right).

<span id="figure-11"></span>

![Figure 11. Interpolating between leftmost and rightmost images with spherical linear interpolation. All samples are generated by a consistency model trained on LSUN Bedroom $256\times 256$.](../../papers/consistency-models/figure-11.png)

**Figure 11.** Interpolating between leftmost and rightmost images with spherical linear interpolation. All samples are generated by a consistency model trained on LSUN Bedroom $256\times 256$.

<span id="figure-12"></span>

![Figure 12. Single-step denoising with a consistency model. The leftmost images are ground truth. For every two rows, the top row shows noisy images with different noise levels, while the bottom row gives denoised images.](../../papers/consistency-models/figure-12.png)

**Figure 12.** Single-step denoising with a consistency model. The leftmost images are ground truth. For every two rows, the top row shows noisy images with different noise levels, while the bottom row gives denoised images.

<span id="figure-13"></span>

![Figure 13. SDEdit with a consistency model. The leftmost images are stroke painting inputs. Images on the right side are the results of stroke-guided image generation (SDEdit).](../../papers/consistency-models/figure-13.png)

**Figure 13.** SDEdit with a consistency model. The leftmost images are stroke painting inputs. Images on the right side are the results of stroke-guided image generation (SDEdit).

<span id="section-12"></span>

## 12 Additional Samples from Consistency Models

We provide additional samples from consistency distillation (CD) and consistency training (CT) on CIFAR-10 ([Figures 14](#figure-14) and [18](#figure-18)), ImageNet $64\times 64$ ([Figures 15](#figure-15) and [19](#figure-19)), LSUN Bedroom $256\times 256$ ([Figures 16](#figure-16) and [20](#figure-20)) and LSUN Cat $256\times 256$ ([Figures 17](#figure-17) and [21](#figure-21)).

<span id="figure-14"></span>

![Figure 14. Uncurated samples from CIFAR-10 $32\times 32$. All corresponding samples use the same initial noise.](../../papers/consistency-models/figure-14.png)

**Figure 14.** Uncurated samples from CIFAR-10 $32\times 32$. All corresponding samples use the same initial noise.

<span id="figure-15"></span>

![Figure 15. Uncurated samples from ImageNet $64\times 64$. All corresponding samples use the same initial noise.](../../papers/consistency-models/figure-15.png)

**Figure 15.** Uncurated samples from ImageNet $64\times 64$. All corresponding samples use the same initial noise.

<span id="figure-16"></span>

![Figure 16. Uncurated samples from LSUN Bedroom $256\times 256$. All corresponding samples use the same initial noise.](../../papers/consistency-models/figure-16.png)

**Figure 16.** Uncurated samples from LSUN Bedroom $256\times 256$. All corresponding samples use the same initial noise.

<span id="figure-17"></span>

![Figure 17. Uncurated samples from LSUN Cat $256\times 256$. All corresponding samples use the same initial noise.](../../papers/consistency-models/figure-17.png)

**Figure 17.** Uncurated samples from LSUN Cat $256\times 256$. All corresponding samples use the same initial noise.

<span id="figure-18"></span>

![Figure 18. Uncurated samples from CIFAR-10 $32\times 32$. All corresponding samples use the same initial noise.](../../papers/consistency-models/figure-18.png)

**Figure 18.** Uncurated samples from CIFAR-10 $32\times 32$. All corresponding samples use the same initial noise.

<span id="figure-19"></span>

![Figure 19. Uncurated samples from ImageNet $64\times 64$. All corresponding samples use the same initial noise.](../../papers/consistency-models/figure-19.png)

**Figure 19.** Uncurated samples from ImageNet $64\times 64$. All corresponding samples use the same initial noise.

<span id="figure-20"></span>

![Figure 20. Uncurated samples from LSUN Bedroom $256\times 256$. All corresponding samples use the same initial noise.](../../papers/consistency-models/figure-20.png)

**Figure 20.** Uncurated samples from LSUN Bedroom $256\times 256$. All corresponding samples use the same initial noise.

<span id="figure-21"></span>

![Figure 21. Uncurated samples from LSUN Cat $256\times 256$. All corresponding samples use the same initial noise.](../../papers/consistency-models/figure-21.png)

**Figure 21.** Uncurated samples from LSUN Cat $256\times 256$. All corresponding samples use the same initial noise.
