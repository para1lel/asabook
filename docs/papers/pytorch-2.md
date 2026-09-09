---
title: 'PyTorch 2: Dynamic Graph Compilation'
createTime: 2026/09/09 12:00:00
permalink: /papers/pytorch-2/
pageClass: paper-reading
---

> **作者:** [Jason Ansel](https://orcid.org/0009-0007-5207-2179), [Edward Yang](https://orcid.org/0009-0008-0621-7872), [Horace He](https://orcid.org/0009-0004-1133-816X), [Natalia Gimelshein](https://orcid.org/0009-0002-9867-5075), [Animesh Jain](https://orcid.org/0000-0001-6777-9168), [Michael Voznesensky](https://orcid.org/0009-0000-0539-0667), [Bin Bao](https://orcid.org/0009-0008-8090-7660), [Peter Bell](https://orcid.org/0009-0003-6824-4343), [David Berard](https://orcid.org/0009-0005-4954-1849), [Evgeni Burovski](https://orcid.org/0000-0001-8149-0483), [Geeta Chauhan](https://orcid.org/0009-0003-0830-7330), [Anjali Chourdia](https://orcid.org/0009-0005-4276-2227), [Will Constable](https://orcid.org/0009-0001-7846-744X), [Alban Desmaison](https://orcid.org/0009-0002-4359-1974), [Zachary DeVito](https://orcid.org/0009-0002-8863-1503), [Elias Ellison](https://orcid.org/0009-0005-8337-3498), [Will Feng](https://orcid.org/0009-0009-6406-4699), [Jiong Gong](https://orcid.org/0009-0009-0845-5628), [Michael Gschwind](https://orcid.org/0009-0001-4963-4915), [Brian Hirsh](https://orcid.org/0009-0004-1239-3320), [Sherlock Huang](https://orcid.org/0009-0005-7558-5570), [Kshiteej Kalambarkar](https://orcid.org/0009-0009-8198-4526), [Laurent Kirsch](https://orcid.org/0009-0007-4121-2308), [Michael Lazos](https://orcid.org/0009-0007-8706-9447), [Mario Lezcano](https://orcid.org/0009-0006-8893-2276), [Yanbo Liang](https://orcid.org/0009-0003-2111-0014), [Jason Liang](https://orcid.org/0009-0008-5462-1466), [Yinghai Lu](https://orcid.org/0009-0003-1993-8648), [C. K. Luk](https://orcid.org/0009-0009-9938-8327), [Bert Maher](https://orcid.org/0009-0004-6873-645X), [Yunjie Pan](https://orcid.org/0009-0002-9351-431X), [Christian Puhrsch](https://orcid.org/0009-0002-3925-967X), [Matthias Reso](https://orcid.org/0000-0002-1582-5860), [Mark Saroufim](https://orcid.org/0009-0009-2612-6588), [Marcos Yukio Siraichi](https://orcid.org/0000-0001-5377-8607), [Helen Suk](https://orcid.org/0009-0007-6048-3189), [Shunting Zhang](https://orcid.org/0009-0008-8370-5554), [Michael Suo](https://orcid.org/0009-0000-5454-3113), [Phil Tillet](https://orcid.org/0009-0007-0636-8710), [Xu Zhao](https://orcid.org/0000-0003-2906-8677), [Eikan Wang](https://orcid.org/0009-0009-2648-5193), [Keren Zhou](https://orcid.org/0000-0002-7977-3182), [Richard Zou](https://orcid.org/0009-0000-9597-1405), [Xiaodong Wang](https://orcid.org/0000-0001-5436-9952), [Ajit Mathews](https://orcid.org/0009-0003-4199-0434), [William Wen](https://orcid.org/0009-0009-1502-9520), [Gregory Chanan](https://orcid.org/0009-0006-0635-4725), [Peng Wu](https://orcid.org/0000-0003-2913-3280), 和 [Soumith Chintala](https://orcid.org/0000-0003-2147-9850).
>
> 2024 年 4 月 27 日发表于 *Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2* (ASPLOS '24), 第 929-947 页. 原文完整标题: [*PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation*](https://doi.org/10.1145/3620665.3640366). [原始 PDF](/paper/pytorch-2.pdf). 本文没有可用的 arXiv 或 TeX 源码; 原始 PDF 是精确印刷版式与参考文献的权威依据.

## 摘要

本文介绍了流行的 PyTorch 机器学习框架的两个扩展 TorchDynamo 和 TorchInductor, 它们实现了 PyTorch 2 中发布的 torch.compile 功能. TorchDynamo 是一个 Python 层即时 (JIT) 编译器, 它让 PyTorch 程序能够进行图编译, 同时不牺牲 Python 的灵活性. 其做法是在执行前动态修改 Python 字节码, 并将连续的 PyTorch 操作提取为 FX 图, 然后使用多个可扩展后端之一对该图进行 JIT 编译. TorchInductor 是 TorchDynamo 的默认编译器后端, 它将 PyTorch 程序转换为面向 GPU 的 OpenAI Triton 代码和面向 CPU 的 C++ 代码. 结果表明, TorchDynamo 能以比以往方法更稳健的方式捕获图, 同时只增加很少的开销; 在 NVIDIA A100 GPU 上的 180 多个真实模型中, TorchInductor 的推理与训练几何平均加速比分别达到 2.27× 和 1.41×, 超过了另外 6 个编译器. 这些扩展为 PyTorch 这类 eager 模式框架提供了一种通过编译器应用优化的新方法.

<span id="section-1"></span>

## 1 引言

现代机器学习框架可分为 PyTorch [Pas19a] 和 JAX [Bra18] 这类 eager 模式框架, 以及 TensorFlow [Aba16c], Caffe [Jia14], Theano [The16] 和 CNTK [Sei16] 这类图模式框架. Eager 模式框架采用命令式的 define-by-run [Tok19] 方法, 其中机器学习模型表示为代码, 每次运行模型时都会执行这些代码. 图模式框架采用更具声明性的 define-and-run [Tok19] 方法, 它们公开一套图构建 API, 要求用户先构建图, 再执行该图.

机器学习框架的用户, 尤其是研究人员, 对 eager 编程模型表现出了压倒性的偏好 [He19a]. Eager 模式模型更容易理解, 也可以使用 Python 中的 print 和 pdb 等标准工具调试 [Inn17]. 用户对 eager 模式的这种偏好促使传统图模式框架转向 eager 模式编程模型 [Agr19].

Eager 模式框架的缺点是难以通过编译器应用图级优化. 框架每次只能看到一个算子, 因而无法自动执行融合或调度等跨越算子边界的优化. 为解决这一问题, 人们曾尝试通过记录/重放 [Dev18a, Ree22], Python 解析 [Dev18a] 和惰性求值 [Suh21] 在 PyTorch 中实现图捕获. 遗憾的是, 这些方法牺牲了很多吸引用户选择 PyTorch 的易用性. 记录/重放并不可靠, 可能产生错误行为 [Dev18a]. Python 解析适用于简单程序, 但一直无法复现完整 Python 的复杂语义, 因而后文结果将表明, 它在一半以上的真实模型上会失败. 惰性求值会产生很高的运行时开销, 并增加内核启动延迟. 此外, 对某些模型而言, 仅使用图模式后端的 PyTorch 并不可行. PyTorch 提供了很大的灵活性, 许多模型作者会使用不易映射到图的功能, 例如字典, 列表, 自定义类, 第三方库 (numpy, logging 等), 磁盘/网络, 多进程, 异常和手写内核.

本文提出了 PyTorch 的两个开源扩展: TorchDynamo 和 TorchInductor. 这两个扩展支撑了 PyTorch 2 引入并于 2023 年 3 月正式发布的 torch.compile 功能. TorchDynamo 是一个 Python 层 JIT 编译器, 旨在让 PyTorch 程序能够进行图编译, 同时保留 Python 的全部灵活性. TorchDynamo 接入 CPython 的 Python 帧求值 API [Vie16], 在 Python 字节码即将执行前对其进行动态修改. 它重写 Python 字节码, 将连续的 PyTorch 操作提取为 FX 图 [Ree22], 再用多个可扩展后端对该图进行即时编译. 它通过字节码分析创建 FX 图, 并生成可与 Python 执行混合使用的较小图片段, 从而兼顾易用性与性能.

TorchInductor 是 TorchDynamo 的新编译器后端. 它将 PyTorch 程序转换为面向 GPU 的 OpenAI Triton [Til19] 代码和面向 CPU 的 C++/OpenMP [Dag98] 代码. TorchInductor 使用与 PyTorch eager 模式相似的抽象, 因而能够支持 PyTorch 的灵活性和动态性. 它引入了一种新的 define-by-run 循环级中间表示 (IR), 便于添加新的算子 lowering. 此外, 它使用 Python 实现, PyTorch 用户因此可以方便地扩展和修改它以满足自己的需要.

实验结果表明, TorchDynamo 能以比以往方法更稳健的方式捕获图, 同时只增加很少的开销. 对于大多数模型, TorchDynamo 能捕获一个完整程序图, 并在必要时平稳回退到部分图. 测量结果表明, TorchInductor 生成的代码平均比另外 6 个 PyTorch 编译器后端更快. 性能比较涵盖训练与推理, CPU 与 GPU, float32 与 float16, 以及 3 个大型基准测试套件, 其中包含来自真实应用的 180 多个完整模型.

<span id="section-2"></span>

## 2 以往的 PyTorch 图捕获尝试

与图模式框架 [Aba16c, Jia14, The16, Sei16] 相比, PyTorch 中的图捕获有其独特挑战; 在图模式框架中, 用户只能使用可在图中表示的结构. 在 PyTorch 和其他 eager 模式框架中, 用户可以在模型内嵌入任意代码, 包括非 PyTorch 库. 因此, 程序经常需要从 PyTorch Tensor 转换为 Python 类型 (通过 .item(), .tolist() 等), 使用外部库 (numpy, logging 等), 以及使用不易映射到固定图抽象的 Python 结构 (类, 闭包, 异常, 控制流等). Python/PyTorch 提供的灵活性与图表示的僵化之间存在错配, 以往在 PyTorch 中捕获图的尝试因而不得不限制用户体验. TorchDynamo 解决了这种灵活性与表示之间的矛盾, 但为提供相关背景, 我们仍会考察这一领域的既有工作.

<span id="section-2-1"></span>

### 2.1 torch.jit.trace

`torch.jit.trace` 使用示例输入进行记录/重放, 生成 TorchScript [Dev18a] 图. 记录发生在 PyTorch dispatcher 层, 该层位于 PyTorch 的 C++ 部分, 用于将算子分派至设备专用内核并支持 autograd. 由于记录在 C++ 中进行, torch.jit.trace 不会捕获 Python 中的任何控制流. 考虑以下示例:

```python
def example1(x):
  if len(torch.nonzero(x)) > 1:
    return x + 1
  return x - 1
```

使用示例输入 `torch.tensor([0, 0])` 时, `torch.jit.trace` 会捕获与下列代码等价的图:

```python
def example1_incorrect_capture(x):
  torch.nonzero(x)
  return x - 1
```

由于程序的执行路径针对示例输入进行了特化, 不同的输入 (例如 torch.tensor([Aba16c, Aba16c])) 会产生错误结果. 此外, 任何非 PyTorch 算子 (例如外部库, 打印, 日志, 副作用等) 都会从捕获的图中省略.

<span id="section-2-2"></span>

### 2.2 torch.jit.script

`torch.jit.script` 也构造 TorchScript [Dev18a] 图, 但它通过解析 Python AST 并执行静态分析来完成. 它能够正确捕获上面的 example1; 与 torch.jit.trace 不同, 这是一种可靠的方法, 不应产生错误结果.

torch.jit.script 面临的主要挑战是, 它试图将完整的 Python 重新实现为静态语言. 这种方法要么全部成功, 要么完全失败: 一旦遇到尚未实现的 Python 组件, 整个程序便无法捕获. 静态模拟完整的 Python 是一项艰巨任务; 实际上, torch.jit.script 只支持 Python 的一个子集. 实验结果表明, torch.jit.script 在 TorchBench 基准测试套件的真实模型上只有约一半时间能够工作; 据我们所知, 有些大型模型需要花数周或数月才能完成 "torchscript" 化, 给用户带来令人沮丧的体验.

<span id="section-2-3"></span>

### 2.3 Lazy Tensors

Lazy Tensors 由 PyTorch/XLA [Pyt23a, Suh21] 项目引入, 该项目主要致力于用 PyTorch 支持 Google TPU [Jou17a]. Lazy Tensors 是一种 C++ 层图捕获技术. 每次迭代时, 它都会推迟操作执行以累积一张图, 然后将累积的图发送给 XLA [Xla17a] 编译器. 通过对该图做哈希, 如果各次迭代捕获的图相同, Lazy Tensors 就能避免重新编译. 这种方法虽然有效且可靠, 但有几个主要缺点:

- 更高的开销: 与 PyTorch eager 相比, Lazy Tensors 需要完成额外工作. 除了运行与 eager 相同的 Python 代码和 PyTorch dispatcher 栈外, 它还必须维护额外的图数据结构, 从而增加运行时成本.
- 引入延迟: PyTorch eager 在模型的第一个操作上就发出第一个内核; 此后, 主机端代码与 GPU 或加速器上的内核并行运行, 从而隐藏开销. 相比之下, Lazy Tensors 要等模型代码执行完才发出第一个内核, 因而会在发出第一个内核之前, 以及任何需要与 CPU 往返的操作之后增加延迟 (这类操作在真实模型中很常见). 因此, Lazy Tensors 常常使主机执行与 GPU/加速器利用串行化, 放大主机端开销. 模型, 损失日志记录和优化器都需要修改才能规避这一问题.
- 重新编译: 每当捕获的图出现新的哈希值时, Lazy Tensors 都必须重新编译. 这可能造成一些频繁重新编译的病态情况.

PyTorch/XLA 项目构建了一个与 TorchDynamo 集成的方案 [Cao22], 混合使用 Lazy Tensors 和 TorchDynamo. 该集成只运行一次 Lazy Tensors 而不是每次迭代都运行, 并使用 TorchDynamo 判断何时需要重新捕获, 从而隐藏 Lazy Tensors 的开销. 本文后面的 PyTorch/XLA 结果使用了该集成.

<span id="section-2-4"></span>

### 2.4 torch.fx.symbolic_trace

`torch.fx.symbolic_trace` [Ree22] 是这些系统中最新的一个, 它引入了与 TorchDynamo 共用的 FX 图格式. 它采用与 torch.jit.trace 相似的基于记录/重放的方法, 但追踪发生在 Python 层, 而不是 PyTorch C++ dispatcher 层. 它使用 Proxy Python 对象运行用户代码以记录其行为, 并使用 PyTorch 中的 torch_function [Abb20] 扩展点. 由于在更高的 Python 层记录, symbolic_trace 能捕获许多 torch.jit.trace 无法捕获的操作. 它使用 Proxy 对象而非真实张量记录, 因而能够检测出许多会使 torch.jit.trace 产生错误结果的情况, 例如尝试从 Proxy 张量中读取大小或值, 或在控制流中使用 Proxy 张量, 如上面的 example1. 它也受前述许多方案要么全有, 要么全无这一限制. 例如, 在上面的控制流情形中, 用户仍被迫重写希望追踪的代码.

遗憾的是, torch.fx.symbolic_trace 仍不可靠, 可能产生错误结果. 考虑以下示例, 它会递增一个全局变量, 并调用一个不依赖函数输入的函数:

```python
def example3(x):
  global call_count
  call_count += 1
  return torch.rand(10) + x
```

如果对该示例运行 torch.fx.symbolic_trace, 它会生成与下列代码等价的图:

```python
def example3_incorrect_capture(x):
  return _tensor_constant0 + x
```

对 torch.rand 的调用被删除, 其结果则以固定常量的形式固化进图中. 后续使用该图时不会获得新的随机性, 而会重复使用捕获期间生成的值. 这种错误捕获可能很难调试, 也可能不被用户察觉. call_count 操作完全丢失, 因为它们没有与 Proxy 对象 x 交互. call_count 只在追踪期间递增到 1, 调用图时不会再递增. 这也是所有图表示都不支持的一类情况. 几乎所有机器学习图格式都没有 Python 全局变量的概念, 因此即使能捕获该操作, 下游后端编译器也不支持它.

<span id="section-2-5"></span>

### 2.5 torch.onnx.export

ONNX [Onn23] 导出其实并不是一种图捕获机制, 但有人会混淆二者, 因此为完整起见, 我们将其列在这里. ONNX 导出在内部使用 torch.jit.trace 和 torch.jit.script ([第 2.1 节](#section-2-1) 和 [第 2.2 节](#section-2-2)), 因而面临这些系统施加的全部限制. 此外, TorchScript 到 ONNX 格式的转换可能失败, 因为 ONNX 并不支持所有 PyTorch 算子. 因此, ONNX 支持的模型集合是 TorchScript 所支持模型集合的子集.

ONNX 团队正在开发与 TorchDynamo 的集成, 以直接集成 TorchDynamo 来取代 TorchScript. 完成后, ONNX 可工作的模型数量将会增加.

<span id="section-2-6"></span>

### 2.6 与 JAX 图捕获的比较

JAX [Bra18] 基本不会遇到 TorchDynamo 所解决的这些挑战. JAX 的初始设计与 XLA [Xla17a] 的设计高度耦合, 而且 JAX 从诞生起便以 XLA 为后端. 这迫使 JAX 程序遵守 XLA 自下而上带入其设计的约束. 因此, JAX 使用一种更简单的捕获机制, 并要求用户编写满足该捕获机制约束的程序. 例如, jax.jit 不支持依赖数据的 Python 控制流, 并要求用户代码满足函数纯度.

相比之下, PyTorch 最初只是一个 eager-only 框架, 其设计中没有任何面向编译器的约束. PyTorch 之上已经积累了大量模型, 其中大多数在编写时完全没有考虑其捕获和编译难度.

从实现层面看, JAX 中的捕获机制与 torch.fx.symbolic_trace ([第 2.4 节](#section-2-4)) 相似, 但略为简单, 因为 JAX 程序是纯函数式的, 不需要考虑状态. Torch FX 论文 [Ree22] 对其与 JAX 的比较有更详细的介绍.

<span id="section-3"></span>

## 3 TorchDynamo 的设计与实现

TorchDynamo 采用了与以往 PyTorch 图捕获系统根本不同的方法. TorchDynamo 不试图删除或替换 Python, 而是通过即时 (JIT) 编译 Python 字节码来配合 CPython 工作. TorchDynamo 是一个从 Python 字节码到 Python 字节码的转换器, 它从原始字节码中提取 PyTorch 操作, 并用对已编译产物的调用替换这些操作; 该产物会融合多个 PyTorch 操作. [图 1](#figure-01) 概述了 TorchDynamo 的工作方式, 本节余下内容将对此加以说明.

<span id="figure-01"></span>

![原始 CPython 帧求值与 TorchDynamo 修改后的帧求值对比图](./pytorch-2/figure-01.png)

**图 1.** TorchDynamo 如何修改 CPython 解释器以捕获 FX 图的概览.

<span id="section-3-1"></span>

### 3.1 使用 API

本文引入的主要 API 是 torch.compile.

它既可以通过对 PyTorch Module 调用来使用, 也可以作为函数装饰器使用. 它提供以下关键字选项:

- backend: 允许用户提供自定义编译函数, 该函数接收一个 torch.fx.Graph 和一个示例输入列表, 并返回一个 Python 可调用对象. 默认值为 TorchInductor, 也可设置为多个内置后端之一或用户定义后端.
- options: 可选的后端专用配置标志字典.
- mode: 一组预定义选项的简写字符串: "default", "reduce-overhead" 或 "max-autotune".

使用 torch.compile 运行模块时, 模块会采用 [图 1](#figure-01) 所示修改后的 CPython 行为执行. 具体而言, 自定义 CPython 帧求值钩子会重写每个正在执行的 Python 函数的字节码, 以提取并编译连续的 PyTorch 操作. 该字节码重写过程会被缓存, 但分析依赖程序的某些动态属性, 后续调用时会用 guard 检查这些属性.

<span id="section-3-2"></span>

### 3.2 CPython 帧求值钩子

PEP 523 [Vie16] 在 CPython 解释器中引入了帧求值 API. 帧是 CPython 用于表示函数调用的数据结构. 这是 TorchDynamo 使用的主要扩展点, 其设计用途就是支持 Python 中的即时 (JIT) 编译器和调试器. PEP 523 向 PyInterpreterState 添加了一个 eval_frame 函数指针, 允许覆写 CPython 中用于解释单次函数调用的核心函数. 每当 CPython 调用函数时, 它会先创建 PyFrameObject, 然后调用这个用户定义的 eval_frame 钩子. 默认情况下, eval_frame 指向 _PyEval_EvalFrameDefault, 后者包含 CPython 的主解释器循环. TorchDynamo 会修改 eval_frame, 以一个对 Python 帧执行 JIT 编译的循环替换这个标准 CPython 解释器循环. TorchDynamo 安装的自定义帧求值函数执行以下操作:

- 检查是否应因文件名排除规则, 此前的分析失败 (这会将帧标记为跳过) 或超出缓存大小限制而跳过该帧. 文件名排除规则用于 Python 标准库和 numpy 等常用库, 这些库不会包含 PyTorch 操作. 对于被跳过的文件, 在原始字节码上调用 _PyEval_EvalFrameDefault 并返回.
- 检查该帧此前是否已编译并缓存; 如果是, 则对缓存中的每个条目执行生成的 guard 函数 ([第 3.3 节](#section-3-3)). 如果某个 guard 函数返回 True, 则使用 _PyEval_EvalFrameDefault 运行匹配的缓存已编译字节码并返回.
- 逐条指令对函数字节码执行符号分析, 以提取 FX 图 [Ree22], guard 和副作用. 如果遇到不支持的操作, 该分析可能在函数中途停止.
- 使用由传给 torch.compile 的 backend= 参数指定的用户定义编译器函数编译 FX 图.
- 生成并编译一个检查所有 guard 的 Python 函数. 如果 guard 均通过且现有已编译产物可复用, 该函数返回 True.
- 如果分析没有到达函数末尾, 则生成 resume_at_XX continuation 函数. Continuation 函数在新帧中运行函数余下部分, [第 3.8 节](#section-3-8) 将介绍这些函数.
- 生成新的 Python 字节码. 该新字节码将: 1) 调用已编译的 FX 图; 2) 存储并重建局部状态/栈状态; 3) 执行原函数本应产生的副作用, 见 [第 3.7 节](#section-3-7); 4) 返回, 或通过回退到原始字节码并调用生成的 continuation 函数来实现 graph break.
- 将生成的 Python 字节码和 guard 函数安装到缓存中, 用 _PyEval_EvalFrameDefault 运行生成的字节码, 然后返回.

<span id="section-3-3"></span>

### 3.3 Guards

Guard 是 TorchDynamo 用于重新检查 JIT 编译所依据的动态属性, 从而判断缓存的编译结果能否复用的机制. TorchDynamo 会为每个经过转换的 PyCodeObject 生成一个 guard 函数, 当复用已编译产物是安全的时, 该函数返回 True. Guard 与转换后的代码均使用 PEP 523 [Vie16] 引入的 _PyCode_SetExtra 扩展点存储. Guard 在分析期间逐步累积, 可以指向来自全局变量/局部变量或嵌套在 Python 数据结构中的变量. 本文写作时共有 30 种不同的 guard. Guard 包括: 检查多种 torch.Tensor 属性, Python 类型, 常量特化, 属性, 字典/列表/元组, nn.Module 实例和 PyTorch 全局状态. Guard 系统横跨 TorchDynamo, AOTAutograd 和 TorchInductor. 任何一层都可以引入 guard 来保护特化. 所有 guard 都是彼此独立的检查, 除去重外互不影响.

<span id="section-3-4"></span>

### 3.4 符号求值

TorchDynamo 的一个基础组成部分是符号 Python 字节码求值器, 它负责分析 Python 字节码并建模每条指令的效果. 符号求值包含跟踪以下内容的数据结构: 1) 栈状态; 2) 局部变量; 3) 异常上下文; 4) 累积的 FX 图 [Ree22]; 5) 累积的 guard; 以及 6) 副作用. 该算法每次处理一条 Python 字节码, 并为每一种 Python 字节码指令类型提供对应的函数.

符号求值开始时, 会检查函数参数并将其转换为符号表示 VariableTracker. 如果字节码访问类属性或全局变量等数据结构, 则会按需添加这些结构的新符号表示. [第 3.5 节](#section-3-5) 会进一步讨论这种表示. 符号求值器从函数的第一条字节码指令开始, 每次处理一条字节码, 直至处理完整个函数. 该分析的正确性可通过归纳法说明: 只要每条字节码都得到正确处理, 整体算法就会正确.

例如, 假设第一条指令是 LOAD_FAST, 这条 Python 字节码会把一个局部变量压入栈中. LOAD_FAST 的处理程序会从符号局部变量中取出变量的表示, 并将其压入符号栈数据结构. BINARY_ADD 的处理程序会从栈中弹出两个符号变量, 然后将结果压回栈中. 结果的计算方式取决于这些变量的类型, 分派方式也会随类型而变化. 如果值表示 PyTorch 张量, 则会向 FX 图 [Ree22] 添加一个新的 add 节点, 并创建一个指向结果节点的新符号张量.

<span id="section-3-5"></span>

### 3.5 Python 数据结构建模

Python 的许多语义存在于库和数据结构中, 因此任何 Python 分析都必须为这些不同类型的行为建模. 为分析每个变量或栈条目的行为, TorchDynamo 提供了一个类层次结构, 用来建模不同数据类型的常见行为. 这些数据结构都是 VariableTracker 的子类. 较重要的 variable tracker 类型包括:

- TensorVariable 表示 torch.Tensor. 它不存储底层张量值, 而是存储一个指向正在部分构建的 FX 图 [Ree22] 的 fx.Proxy, 以及一个表示张量元数据而不含实际数据的 "fake" 张量 (见 [第 5 节](#section-5)).
- ConstDictVariable 和 DataClassVariable 用于表示键值对, 其中键为常量字符串, 值可以是任何内容, 包括嵌套字典/列表.
- ListVariable 和 TupleVariable 表示列表/元组, 可以包含任何其他类型的符号变量.
- UserFunctionVariable 和 UserMethodVariable 表示可被内联的用户定义函数. 它们还支持动态构造且含有闭包的函数.
- UserDefinedClassVariable 表示用户定义类, UserDefinedObjectVariable 表示其实例. 当访问这些对象的属性时, 我们会按需对其特化, 并跟踪它们发生的变更 ([第 3.7 节](#section-3-7)).

还有许多其他 variable tracker 类型用于表示其他情况. 除类型专用数据外, 每个 VariableTracker 实例还包含一组 guard, 这些 guard 在实例创建时初始化, 并通过并集在操作间传播. 此外, 每个实例还会跟踪自己的来源, 以便输出字节码可以加载或修改它.

<span id="section-3-6"></span>

### 3.6 内联, 控制流与闭包

函数调用既可以直接来自用户代码, 也可以通过 __getitem__ 等魔术方法隐式发生. 为收集更大的图, TorchDynamo 会尝试内联函数调用并展平程序. 遇到函数调用时, TorchDynamo 首先为当前符号状态创建检查点. 接着, 它以递归方式尝试对被调用函数执行符号求值, 传入所有输入符号状态并记录发生的所有变化. 如果该递归分析遇到会导致 graph break ([第 3.8 节](#section-3-8)) 或其他错误的情况, TorchDynamo 会回滚到函数调用前的符号状态, 并在该函数调用处生成 graph break. 否则, 递归分析返回, 并继续分析父函数. Python 字节码中的大多数控制流都会通过特化优化掉并得到处理. 例如, 迭代 torch.nn.Module 列表时, TorchDynamo 会用 guard 确认列表不变, 并展开循环. 对于基于张量类型, 大小和形状的控制流, TorchDynamo 会对这些属性设置 guard 并删除控制流. 在控制流无法删除的较少见情况中 (例如基于张量值而非元数据分支), TorchDynamo 会生成 graph break, 让该分支字节码在 CPython 中运行, 并在跳转后恢复分析.

闭包是另一个挑战. 考虑以下示例:

```python
def closure_example(x):
  y = torch.sigmoid(x)
  return lambda z: y + z
```

这里的变量 𝑦 位于闭包中, CPython 使用称为 cell 的对象表示它; cell 增加了一层间接访问, 使闭包中的变量可以修改. TorchDynamo 必须处理多种不同的闭包情况:

- 在捕获区域外创建的 cell 变量必须以不同于其他变量的方式访问. 如果从顶层函数访问这些变量, 可以生成 LOAD_DEREF 和 STORE_DEREF 字节码来访问. 内联时不能使用这些字节码, TorchDynamo 会改为生成直接读写内联函数 cell 的代码, 例如 fn.__closure__[0].cell_contents. 如果 cell 内容发生变化, TorchDynamo 会以跟踪其他变更相同的方式跟踪该变化 ([第 3.7 节](#section-3-7)).
- 在捕获区域内创建并销毁的 cell 变量最容易处理, 也最常见. 这种情况下, TorchDynamo 会通过静态优化消除闭包.
- 在捕获区域内创建但逃逸出帧的 cell 变量最难处理. 这种情况下, TorchDynamo 会优化掉捕获区域内闭包的所有用法. 然后, 在生成字节码的最后, 它会创建所有需要返回的 cell 和 Python 函数对象. 从外部看, 调用者无法分辨返回的闭包是否以不同于原程序的方式创建.

<span id="section-3-7"></span>

### 3.7 变更与副作用

Python 函数有时会产生副作用. TorchDynamo 会将副作用推迟到 FX 图 [Ree22] 调用之后处理, 再生成在最后应用所有副作用的输出字节码. 为此, TorchDynamo 使用一个副作用数据结构来跟踪原始代码本应产生的全部副作用. 如果代码试图读取本应被待处理副作用修改的值, 则会读取对应的待定值. 图生成后, 垃圾回收过程会删除没有逃逸出分析上下文的副作用, TorchDynamo 随后生成应用所需副作用的输出代码. 以这种方式处理副作用会将对同一个值的多次写入合并为一次写入. TorchDynamo 支持以下类型的副作用:

- 如果目标全局变量在同一个文件中, 对全局变量的写入会产生一条 STORE_GLOBAL 字节码. 如果它在不同文件中 (这是内联造成的), 则生成代码来修改另一个模块中的全局变量.
- 对属性 (例如类的属性) 的写入以类似方式处理, 并映射为输出字节码中的 STORE_ATTR. 我们使用 VariableTracker 上的 source 判断如何加载对待修改对象的引用.
- 对 cell/闭包的写入会被跟踪, 并以多种方式处理 (见 [第 3.6 节](#section-3-6)).
- 构造类时, 会创建一个占位符符号对象, 内联 __init__ 方法, 并跟踪该占位对象上的所有属性变更. 如果函数结束时该对象仍存活, 输出字节码将创建该对象 (绕过构造函数) 并设置所需属性.
- 如果字典/列表作为输入传入或从全局变量/属性加载, 对其进行修改也可能产生副作用. 字典/列表的 VariableTracker 表示会对这些对象的初始符号状态设置 guard, 然后在整个函数中以符号方式跟踪所有变化. 捕获的 FX 图 [Ree22] 会优化掉所有这些操作. 输出字节码将创建与最终状态相符的新字典/列表, 并修改原列表对象以匹配该对象. 对于没有逃逸出捕获区域的列表/字典, 不需要重建, 因为其变更无法被观察, 所以可以完全删除.

<span id="section-3-8"></span>

### 3.8 Graph break 与 continuation 函数

当 TorchDynamo 遇到无法处理的 Python 字节码时, 例如对外部库的调用, 它会生成所谓的 graph break, 将正在分析的字节码分成多个片段. 实质上, TorchDynamo 会把已编译片段混入原始 Python 代码, 得到混合执行. 所有待处理的部分 FX 图 [Ree22] 都会被编译. 在输出代码中, 部分图调用后会执行不受支持的字节码, 随后我们将递归使用 TorchDynamo 分析函数余下部分. 为触发这种递归分析, TorchDynamo 会生成一个或多个以下形式的 continuation 函数:

```python
def resume_at_X(... livevars ...):
  ... restore try/except/stack state ...
  JUMP_ABSOLUTE X
  ... original function bytecode ...
```

该 continuation 函数看起来与原始函数很相似, 但有几处变化: 1) 参数经过修改, 以反映跨越 graph break 仍存活的变量; 2) 添加一个前缀来恢复栈/异常状态, 这些状态也可以作为参数传入; 3) 创建一条 JUMP_ABSOLUTE 指令, 使执行从函数中间恢复.

TorchDynamo 会生成一个这样的函数; 如果存在控制流 (所有控制流字节码都恰好有两个分支), 则会生成两个, 以便在不受支持的字节码之后继续执行. 将 continuation 组织为 Python 函数的优点是, 它会通过帧求值 API 递归触发 TorchDynamo. TorchDynamo 处理 continuation 函数时, 会将其视为与其他 Python 函数完全相同的函数.

<span id="section-3-9"></span>

### 3.9 AOTAutograd

AOTAutograd 是 PyTorch 中的可复用组件, 许多 PyTorch 编译器后端调用它来增加训练支持并使用共享的算子分解. TorchDynamo 捕获模型的前向过程, 但为支持训练, 还需要生成反向传播过程. 在 PyTorch eager 中, 反向图使用基于 tape 的 autograd [Pas19a] 动态生成. AOTAutograd 以支持部分程序图的方式将前向图转换为前向图和反向图.

AOTAutograd 通过在 fake tensor 输入上运行 PyTorch eager 模式 autograd 引擎, 记录前向与反向联合图. 依赖数据的操作无法在 fake tensor 上工作 (因为没有底层数据), 因此我们在 TorchDynamo 中针对这些操作触发 graph break, 并在图外运行它们. 随后, AOTAutograd 使用 min-cut 算法 [Yu23b] 将该联合图分割为独立的前向图和反向图, 并优化内存使用. 作为 min-cut 算法的一部分, 我们应用后端专用优化, 在反向图中重新物化某些重新计算成本较低的激活值.

AOTAutograd 还会对图应用其他 dispatcher 层转换. 在分解过程中, AOTAutograd 将一些 PyTorch 算子映射到一组更小, 更基础的算子. AOTAutograd 也会删除执行变更的操作并用其函数式等价形式替换, 使图变为纯函数式.

<span id="section-4"></span>

## 4 TorchInductor 的设计与实现

TorchDynamo 解决了 PyTorch 中的图捕获问题, 但要发挥作用, 它必须与后端编译器配合; 该编译器需要接收捕获的 FX 图 [Ree22], 并从中生成快速代码. 我们创建 TorchInductor 作为参考编译器后端. 它被设计为通用编译器, 既可由用户直接使用, 也可作为其他后端的起点.

<span id="section-4-1"></span>

### 4.1 设计原则与关键技术

在深入讨论 TorchInductor 的设计前, 先介绍促成其设计的一些原则与技术:

**PyTorch 原生:** PyTorch 做出了许多不同于其他框架和编译器的设计选择: Tensor 的 stride 对外公开, 用户可加以操作; aliasing view 很常见; 数据与元数据均可原地修改. 任何采用截然不同计算模型的编译器都会在表示 PyTorch 程序时遇到许多挑战. 我们希望 TorchInductor 采用与 PyTorch eager 相似的抽象, 通过一层轻量转换来支持完整的 PyTorch.

**Python 优先:** 大多数 PyTorch 用户最熟悉 Python. PyTorch 的 Python 部分得到的社区贡献远多于 C++ 部分. 我们选择用 Python 实现 TorchInductor, 让 PyTorch 用户容易理解和改造它.

**广度优先:** 我们没有专注于已得到充分研究的一小组模型 (如 ResNet/BERT), 而是有意在早期将重点放在支持多种算子, 硬件和优化上. 这使 TorchInductor 成为可扩展至多种场景的通用编译器. 这也是早期重点放在训练上的原因, 因为训练对编译器而言是比推理困难得多的问题.

**复用先进语言:** 在选择输出语言时, 我们参考了 PyTorch 用户编写高性能内核的方式. 我们观察到, 用于编写 GPU 内核的 OpenAI Triton [Til19] DSL 正迅速普及, 而这些内核的性能常常超过其他编译器和先进库. 高性能 CPU 内核通常使用 C++/OpenMP [Dag98] 编写. TorchInductor 同时生成 Triton 和 C++ 输出代码, 因此既能利用这些项目的技术, 也能生成 PyTorch 用户易于理解的输出代码.

<span id="section-4-2"></span>

### 4.2 分解

TorchInductor 不会为 PyTorch 中的所有算子实现到自身 IR 的 lowering, 而会将许多 PyTorch 算子分解为一组更容易处理的简单算子. 这些分解通过 AOTAutograd ([第 3.9 节](#section-3-9)) 完成, TorchInductor 调用它时会传入所需分解的字典. 分解以一个使用其他算子实现 PyTorch 算子的 Python 实现来编写; 例如, 下列代码将 `log2` 分解为 `log` 和 `mul`:

```python
log2_scale = 1 / math.log(2)

@register_decomposition(torch.ops.aten.log2)
def log2(x):
  return torch.log(x) * log2_scale
```

该分解将被递归追踪并规范化, 过程中可能触发更多分解, 直至到达不动点. 注意, 活跃的分解集合不能包含环. 本文写作时, TorchInductor 使用了 191 个分解 (计入重载后为 387 个). 这些分解大多并非 TorchInductor 专用, 任何其他后端都可通过 torch._decomp 模块使用; 另有一些是 TorchInductor 专用分解.

<span id="section-4-3"></span>

### 4.3 Lowering 与 define-by-run 循环级 IR

编译的下一阶段是将 PyTorch 操作的 FX 图 lowering 为 TorchInductor 的 define-by-run IR. Define-by-run IR 指 IR 使用可执行 Python 代码定义循环体, 这使 TorchInductor IR 拥有完整 Python 的大部分能力, 无需大量样板代码, 也能简洁地编写 lowering. Lowering 通过对 FX 图进行符号解释, 并应用负责转换单个算子的 lowering 函数来完成. 本文写作时, TorchInductor 为 433 个 PyTorch 算子提供了 lowering (计入重载后为 1605 个). 如果遇到未知算子, 它会被自动转换为运行原始 PyTorch 代码的 fallback kernel 节点.

<span id="figure-02"></span>

![二维 log2 操作的 TorchInductor 中间表示](./pytorch-2/figure-02.png)

**图 2.** 二维张量上 `torch.log2` 的 TorchInductor IR.

[图 2](#figure-02) 所示示例 IR 中, inner_fn_buf0 是一个 Python 函数, 它使用对 ops.* 命名空间中 TorchInductor 基础算子的调用, 定义如何计算张量 buf0 的单个元素. 该函数接收一个 SymPy [Meu17] 符号列表 (i0 和 i1), 表示待计算元素的符号坐标. SymPy 符号 s0 和 s1 表示待计算张量的大小, 同时用于 size 和 stride. 这些 size 符号被捕获在 Python 闭包中, 并注册到图对象上.

TensorBox 和 StorageBox 是与 PyTorch torch.Tensor 和 torch.Storage 对象对应的抽象, 用于在 lowering 过程中处理 view, aliasing 和变更. ComputedBuffer 表示将由生成代码计算的张量 (区别于通过 fallback kernel 创建的张量或输入). Pointwise 表示 ComputedBuffer 是数据并行的逐点计算. IR 还支持 Reduction 和 Scatter, 用于处理其他类型的算子.

该 IR 的主要优点是, 借助 Python 的全部能力, 它很容易构造. 用户可以组合不同的 IR 节点, 并在其中嵌入逻辑. 上面的示例最初不会构造为单个扁平函数, 而是在 lowering 过程中定义为许多较小的函数闭包. 为 ops.mul 创建的函数将调用为 ops.log 创建的另一个函数, 后者再调用另一个为加载输入参数而创建的函数.

编译和分析该 IR 的方式以 ops.* 的虚拟化命名空间为基础, 该命名空间可以动态覆写以执行不同功能. 为分析该 IR, 我们让 ops 指向一个分析 pass, 它可以执行记录内存访问或记录用于强度削减优化的高/低水位等操作. 为使用该 IR 生成代码, 我们让 ops 指向写出 Triton 或 C++ 代码的对象. 为转换该 IR, 我们使用 FX 追踪, 以访问这些 Python 函数的图表示.

本文写作时, TorchInductor 循环级 IR 包含 54 个基础算子:

- ops.load 和 ops.store 使用给定的 buffer 名称与指定符号内存位置的 SymPy index 来访问 Tensor 内存.
- ops.reduction 的操作方式类似 ops.store, 但 reduction 隐式发生在写入内部. 它使用给定的 reduction 类型, 沿当前节点的 reduction 维度合并存储值. 支持的 reduction 类型包括: argmin, argmax, any, max, min, prod, sum, xor_sum 和 welford_combine [Wel62].
- ops.index_expr 将用于索引的 SymPy 表达式转换为用于计算的值.
- ops.indirect_indexing 通过引入一个动态绑定的新 SymPy 变量, 将计算值转换为用于索引的 SymPy 表达式.
- ops.masked 实现条件执行. 它接收一个条件以及一个没有参数的 Python 函数 (递归使用相同 IR). 它会映射为 Triton 中的 mask 和 C++ 中的条件语句.
- ops.load_seed, ops.rand, ops.randn 和 ops.randint64 用于计算随机数.
- 其余 ops 为逐元素数学操作.

<span id="section-4-4"></span>

### 4.4 调度

TorchInductor 的调度阶段决定融合哪些算子, 内核以何种顺序运行, 并为删除和/或复用 buffer 制定内存规划. 调度首先将 IR 中的每个 buffer 转换为 BaseSchedulerNode 的一个子类. SchedulerNode 表示由 TorchInductor 生成主体代码的标准内核. ExternKernelSchedulerNode 表示对库代码或用户定义内核的调用. 此外, NopKernelSchedulerNode 不映射到任何内容, 但用于添加依赖边, 以确保内核顺序 (例如 concatenate 内核, 其处理方式是让 producer 直接写入合并后的 buffer). 最后, FusedSchedulerNode 表示一组融合为单个内核的两个或更多 SchedulerNode.

接下来, 调度器会把每个内核的内存读写集合转换为节点间的依赖边. 依赖边带有被读取符号内存地址的标注. 符号内存地址对于判断哪些融合合法十分重要. 例如, 如果一个内核以前向顺序写入 buf0, 但 consumer 以逆向顺序读取 (使用 ops.load("buf0", s0 -1 -i0)), 则这些节点不能融合.

融合由两个关键函数控制:

- Scheduler.can_fuse(node1, node2) 在两个节点可以融合时返回 True. 该函数检查依赖边, 还会检查许多其他属性以保证融合正确. 其中也包含一些启发式规则; 例如, 如果 config.aggressive_fusion=False, can_fuse 将阻止不共享任何内存访问的节点融合. 这里也有后端专用逻辑; 例如, TorchInductor 支持 Triton 的 reduction-broadcast-reduction 融合, 但不支持 C++ 的这种融合.
- Scheduler.score_fusion(node1, node2) 用于排列不同融合可能性的顺序. 某些融合互斥, 因此 TorchInductor 会选择得分较高者. 融合得分按照以下因素排序: 1) 融合类别 (如 pointwise/reduction/template); 2) 该融合预计节省的内存流量字节数; 以及 3) 原图中节点之间较短的距离. TorchInductor 会在循环中执行以下贪心算法, 直至不再有其他融合 (因为某些融合可能创造更多融合机会): 1) 找出所有融合机会; 2) 为每个融合机会打分并按得分排序; 3) 对每个融合机会检查该融合是否仍然合法, 若合法则应用. 两个节点融合后, 所有指向组成节点的待处理融合机会都会更新为指向新的融合节点.

<span id="section-4-5"></span>

### 4.5 Triton 代码生成

<span id="figure-03"></span>

![为 log2 示例生成的 Triton 内核](./pytorch-2/figure-03.png)

**图 3.** 为 [图 2](#figure-02) 生成的 Triton 代码.

Triton 代码生成负责将 TorchInductor IR 映射为输出 Triton [Til19] 内核. [图 3](#figure-03) 展示了为上面的 log2 示例生成的代码. 该内核每次处理一个由 XBLOCK 个元素组成的块. 如果元素数量不是 XBLOCK 的倍数, 末尾的一些元素可能被屏蔽. 我们在代码生成期间简化索引. 例如, 本例会把 IR 中的二维 strided load 转换为 contiguous load. 代码生成还负责公共子表达式消除 (CSE), 其做法是在打印代码行并分配以 tmp 开头的中间变量名时使用缓存. Pointwise 装饰器对用于支持块大小启发式规则, 自动调优和提前内核编译的样板代码进行编码. 装饰器表示正在生成的内核类型 (pointwise, reduction 或 template), 其参数则是数据对齐等内核必需的元数据. 生成 reduction 内核时, TorchInductor 有两种代码生成模式. 对于较小的 reduction, 它会生成 persistent reduction, 将整个 reduction 加载到单个块中并保留在寄存器/共享内存中; 此时 reduction 直接映射到 Triton reduction 算子. 对于较大的 reduction, TorchInductor 会生成一个循环, 使用整个块作为 accumulator, 并在循环末尾调用 Triton reduction. 对于更复杂的操作 (矩阵乘法和卷积), TorchInductor 提供自己的 Triton 代码模板系统, 将手写 Triton 与生成的 Triton 混合使用. 模板使用 Jinja [Mon23a] 编写, 并通过辅助方法与 TorchInductor 的代码生成系统交互.

<span id="section-4-6"></span>

### 4.6 C++ 代码生成

对于 CPU 后端, TorchInductor 生成带 OpenMP [Dag98] 的 C++ 代码. C++ 后端内有两个变体, 即向量化变体和非向量化变体. 向量化变体执行 tiling, 并将大多数操作映射到 PyTorch 源码中包含的 `at::vec::Vectorized` 类. 该类每次处理 16 个元素, 与标准 PyTorch 内核的向量化方式相同, 并支持多种 SIMD 指令集. 非向量化变体使用许多 C++ 标准模板库 [Pro98] (STL) 函数生成较为标准的 C++ 代码. 两种变体都使用 #pragma omp for 标注进行并行化, 并通过一些启发式规则决定并行化多少层循环. 如果 reduction 维度循环已并行化, reduction 会映射到 OpenMP reduction 标注; 否则映射到带 accumulator 的 C++ 循环.

<span id="section-4-7"></span>

### 4.7 Wrapper 代码生成

Wrapper 代码生成负责生成调用 Triton, C++ 和外部来源内核的代码. 它还会计算张量大小, 并处理内存分配与释放. Wrapper 代码生成有两种不同实现, 一种生成 Python 代码, 另一种生成 C++ 代码. Python 后端更灵活, 支持 C++ 后端不支持的一些边缘情况; C++ 后端的开销更低.

以 mode="reduce-overhead" 启用时, TorchInductor 使用 CUDA Graphs [Gra19] 完全消除 wrapper 代码的开销. CUDA Graphs 在 CUDA driver 层记录并重放内核启动, 开销甚至低于 C++ wrapper 代码. 为保证可靠性, CUDA Graphs 只在满足安全要求时使用, 并会在某些情况下自动禁用 (例如存在动态形状, 非 CUDA 张量等情况).

<span id="section-4-8"></span>

### 4.8 相关深度学习编译器

深度学习编译器领域有许多令人期待的工作. 大多数 PyTorch 用户使用 GPU, 因此我们选择 Triton [Til19] 作为输出目标的主要原因是: 已经证明, 用简单输入代码即可生成性能超过手写库 [Nvi23g, Che14, Cut23] 的内核. 极少有编译器能稳定做到这一点, 而许多被广泛使用的深度学习编译器只是直接调用这些库, 并不试图在复杂内核的 GPU 代码生成上与之竞争.

许多编译器采用受 Halide [Rag13] 启发的设计, 包括 TVM [Che18e], nvFuser [Sar22] 和 NNC [Zol21]. 这些设计将语义语言与调度语言分开, 因而可以在不改变程序语义的情况下探索不同调度. 研究人员探索了多种不同的搜索空间表达方式 [Din23c, Sha22b, Vas18, Wen21, Zhe22d, Bag19, Zhu22, Fen23], 也研究了如何自动搜索该空间 [Zhe20, Che18a, Xin22, Zhe22c, Ans14].

XLA [Xla17a] 是 TensorFlow [Aba15] 和 JAX [Bra18] 背后的编译器. XLA 提供多层 IR, 包括已经成为 TPU [Jou17a] 和类似加速器标准的高层 IR HLO. MLIR [Lat21] 生态系统中正在涌现许多较新的编译器, 包括 IREE [Ire19] (现为 OpenXLA [Xla17a] 的一部分). 最新版 Triton [Til19] 的内部表示也使用 MLIR.

<span id="section-5"></span>

## 5 动态形状

深度学习编译器通常只能处理静态形状, 也就是说, 它们生成的已编译程序仅适用于一种特定的输入形状配置, 任何输入形状发生变化时都必须重新编译. 这一假设很适合今天运行的大多数常见深度学习模型, 但在以下几种情况下并不充分:

- 某些维度可能变化, 例如 batch size 或 sequence length. 例如, 执行自适应批处理的推理服务会根据其批处理时间窗口内收到的请求数, 以不同的 batch size 执行推理请求. 我们也可能希望只将变长序列填充到一个 batch 内的最大 sequence length, 而该值可能随 batch 变化.
- 某些模型的输出形状依赖数据, 也就是说, 输出和中间结果的大小可能取决于实际输入数据, 后者可能在各次运行间变化. 例如, 检测模型可能先生成数量不定的候选 bounding box, 再运行成本更高的图像识别模型, 判断目标是否位于某个 bounding box 中. Bounding box 的数量取决于数据.
- 处理 sparse tensor, jagged tensor 和图神经网络等稀疏表示时, 会出现一种尤为重要的数据依赖形状. 在所有这些情况中, 待处理的数据量取决于问题的稀疏结构, 通常会以依赖数据的方式变化.

在支持动态形状时, 我们选择不支持动态 rank 程序, 例如输入张量的维数发生变化的程序, 因为真实深度学习程序中很少出现这种模式; 这也不必再对形状的符号列表进行归纳推理.

<span id="section-5-1"></span>

### 5.1 符号形状 guard

TorchDynamo 使用直线 trace 的出发点, 是为了复用现有的面向 PyTorch API 编写的 Python/C++ 代码. 对于动态形状, 我们延续了这一思路: 与可能同时捕获条件语句两个分支的完全符号化系统不同, 我们始终选择一个分支, 并在假定只有这些假设成立时才能复用该 trace 的前提下进行特化. 为此, 我们为每个符号 size 维护一项 size hint, 表明第一次触发即时编译的输入中该 size 的具体值. 对张量形状执行条件判断时, 我们查询该 hint 来决定选择哪个分支, 并添加 guard.

这会显著简化我们生成的符号形状公式, 因为无需表示条件语句, 但也意味着 guard 管理系统要复杂得多. 例如, 考虑以下程序: def f(x, y): z = torch.cat([x, y]) if z.size(0) > 2: return z.mul(2) return z.add(2)

我们最终用 TorchInductor 编译的 IR 将是 torch.cat([x, y]).add(2) 或 torch.cat([x, y]).mul(2) (条件已被展平删除), 但要判断进入哪个分支, 必须知道中间结果 z 的大小. 由于 TorchDynamo 必须预先知道编译后的 trace 是否有效 (我们不像某些 JIT 编译器那样支持 bailout), 必须能够将 z.size(0) 化简为关于输入的表达式 x.size(0) + y.size(0). 这通过为 PyTorch 中的所有算子编写 meta 函数完成. Meta 函数将 size 信息传播到张量输出, 而不实际对节点执行计算. 本文写作时, meta 函数覆盖 3028 个 PyTorch 算子中的 2657 个 (计入重载); 由于很少或从未使用的算子形成了一条长尾, 这一覆盖范围包含绝大多数真实模型. 此外, 还有一种机制可为自定义算子定义自己的 meta 函数.

<span id="section-5-2"></span>

### 5.2 优化动态形状推理

动态形状的一个主要动机是缩短编译时间, 因为只支持静态形状的编译器必须针对输入形状的每种可能组合重新编译内核. 然而, 对符号形状推理本身也有成本: 极端情况下, 输出张量的形状表达式可能相当复杂. 我们采用多种策略来减小符号形状推理对性能的影响:

- 动态形状的默认 API 不要求任何用户标注: 我们假定所有输入都可能是动态的, 模型权重是静态的, 并通过逐步执行模型, 分析二者间的交互来推断真正的动态性. 我们还支持 assume_static_by_default 模式, 它强制假定所有输入维度均为静态, 除非用户使用 mark_dynamic(tensor, dim) 明确标记其为动态维度.
- PyTorch 代码经常测试某个变量的 size 是否为 0 或 1; 例如, 构造张量时, PyTorch 会计算其是否连续. 零元素张量始终连续, 所以我们总会测试张量的每个维度是否为 0. 我们不会迫使符号推理系统在每次 trace 时重新发现这一事实, 而是主动执行 0/1 特化: 如果输入 size 为 0 或 1, 不给它分配符号变量, 而将其视作常量并添加适当的 guard. 对 1 特化对于捕获 PyTorch 中的广播语义和性能优化很重要. 重要的是, 分配符号变量时, 我们可以做出否定推断: 任何符号变量都必定不等于 0/1, 因此测试它是否等于 0/1 时, 无需引入额外 guard 即可将表达式求值为 false.
- 处理用户程序时, 我们会随着从 guard 获得更多事实, 逐步简化符号表达式. 当前实现会即时简化合一和整除性, 还会使用 SymPy [Meu17] 帮助判断所请求的 guard 是否已经可以静态确定; 如果是, 则可将其消除.

<span id="section-5-3"></span>

### 5.3 无 hint (无底层值) 的符号整数

为解析控制流, 我们会检查符号整数的实际值以决定选择哪个分支, 并为其设置 guard. 当某个 size 变量来自 .nonzero() 或 .item() 等依赖数据的操作, 且实际值未知时, 就会产生无底层值的符号整数. 对这些符号整数执行控制流是非法的, 因此必须在这些操作处触发 graph break. 如果直接实现, 这一限制过于严格, 会造成太多 graph break. 为解决这些问题, 最重要的增强包括: 1) 创建张量时, PyTorch 会预先计算张量的相关数据; 例如, 使用 empty_strided 创建张量时, 它会对 stride 排序, 并判断张量是否不重叠且稠密. 排序会产生大量 guard. 然而, 更常见的做法是使用 empty 等高层 API 直接生成张量, 这种方式保证生成不重叠且稠密的张量. 我们修改了 PyTorch, 以免无谓地重新计算这些属性. 2) 即使需要进行非平凡计算, 某项属性有时也不会被使用. 将这些预计算属性设为惰性, 可以避免为未使用的属性设置 guard. 3) 通常无法确定整数张量中的数据是否可能为负. 不过, 我们提供 constrain_range API, 用户可用它指定 size 的上下界均受已知范围限制.

<span id="section-6"></span>

## 6 实验结果

我们的评估使用 3 个不同的基准测试套件. TorchBench [Con20] 包含一组多样的模型, 它们取自开源仓库, 并根据 Papers with Code [Sar21] 对高引用项目的排名加以选择. HuggingFace [Wol19] 是一个流行的 Transformer [Vas17d] 模型库. TIMM [Wig19] 是一个流行的 PyTorch 视觉模型库. 为将后两个库转为基准测试套件, 我们选择了覆盖所有现有模型类别的代表性模型.

我们的基准测试基础设施已开源 [Tor23], 希望其他论文也能使用它. TorchInductor Performance Dashboard [Tor23a] 中提供更多结果, 包括单模型性能, 不同 TorchInductor 设置, 以及 PyTorch nightly build 的每日更新. 实验在 NVIDIA A100 GPU, CUDA 11.6 和 Intel Xeon 8275CL CPU 上运行. 为减小噪声, 实验重复 100 次, 并进行 3 次 warm up 迭代. 我们对每个模型设置 30 分钟超时, 并将超时计为失败. TorchInductor 使用 2023 年 8 月 30 日的 PyTorch nightly build 运行, 并启用 max-autotune, freezing 和 cudagraphs. 其他版本为: nvFuser 2.0; NNC 2.0; Hidet 0.2.2; TVM 0.11.1; ONNX Runtime (ONNXRT) 1.14.1; 以及 PyTorch/XLA 2.1. 在训练实验中, 我们测量前向和反向传播一次 step 的时间, 不含优化器.

<span id="section-6-1"></span>

### 6.1 TorchDynamo 捕获图的能力

[表 1](#table-01) 第一部分给出了实验结果, 比较 TorchDynamo 和 TorchScript [Dev18a] 捕获不同基准测试套件的能力. 对 HuggingFace 而言, TorchScript 在每个模型上都会失败, 因为 HuggingFace 模型返回 TorchScript 不支持的 ModelOutput 容器类. 大多数 TIMM 模型可以与 TorchScript 配合工作, 因为 TIMM 的维护者在工作流程中使用 TorchScript, 并投入精力适配这些模型. 在 TorchBench 上, TorchDynamo 可以工作的模型数量超过 TorchScript 的两倍. TorchBench 由来自不同来源的模型组成, 因此在 3 个基准测试套件中最能代表图捕获比较.

[表 1](#table-01) 第二部分给出了 TorchDynamo 捕获图质量的统计数据, 并以可工作模型的百分比进行归一化. 与以往要么全部成功, 要么完全失败的系统不同, TorchDynamo 可以捕获部分程序和多张图. TorchDynamo 大多数时候能够捕获单个完整程序图, 即使发生 graph break, 典型图的大小也有数百个算子. Graph break 最常见的原因是: 使用 numpy [Har20b] 等非 PyTorch 库; 转换为 tolist() 等 Python 类型; 以及依赖数据的控制流操作. torch.compile 支持编译 numpy 操作, 但本实验没有启用该功能.

<span id="table-01"></span>

![TorchBench, HuggingFace 和 TIMM 上的 TorchDynamo 图捕获统计](./pytorch-2/table-01.png)

**表 1.** 各基准测试套件的 TorchDynamo 统计数据, 使用 NVIDIA A100 GPU 上的 float32 推理测得.

<span id="section-6-2"></span>

### 6.2 图捕获开销

<span id="table-02"></span>

![TorchDynamo 与 Lazy Tensors 的图捕获开销](./pytorch-2/table-02.png)

**表 2.** 图捕获开销 (越低越好), 以 eager PyTorch 执行时间的百分比表示. 本实验使用与 eager PyTorch 相同的内核, 因此只计图捕获成本. 在 NVIDIA V100 GPU 上使用 float32 TorchBench 测得.

[表 2](#table-02) 测量了 TorchDynamo 和 Lazy Tensors 由图捕获引入的运行时开销. 其他系统在提前阶段运行, 不会引入运行时开销. 我们让每个系统使用与 PyTorch eager 相同的内核, 因此减速仅来自图捕获开销. 我们计算 TorchBench 上减速比的几何平均值, 再减去 1, 得到增加开销的百分比. 与所有结果一样, 计时中排除了 warm up 迭代, 因此测量的是稳态性能.

TorchDynamo 的开销低于 5%, Lazy Tensors 则会增加大量开销. 这些 Lazy Tensor 开销在不同模型间并不均匀. 对于采用跨迭代流水线的训练: 三分之一模型的开销低于 10%, 三分之一模型的开销介于 10% 和 66%, 另三分之一模型的开销介于 66% 和 1759%.

减轻 Lazy Tensor 在训练和离线推理中开销的一种方法是跨迭代流水线. 这有助于解决以下问题: 在 Lazy Tensors 的单次迭代中, CPU 捕获时 GPU 空闲, GPU 执行捕获内容时 CPU 空闲. 运行多次迭代即可使迭代 𝑁 的捕获与迭代 𝑁 -1 的执行重叠. [表 2](#table-02) 中 Lazy Tensors + 跨迭代流水线通过测量 10 次迭代而非 1 次迭代, 衡量这种摊销效果. 该策略能小幅改善 Lazy Tensor 的开销.

对许多模型而言, Lazy Tensor 捕获速度太慢, 无法使 GPU 饱和. 这一点在较小模型或含有大量操作的模型上尤其明显. 这时流水线没有帮助, 因为限制因素是 Lazy Tensor 开销. 某些 PyTorch 模型包含 if torch.any(torch.isnan(x)) 或 print(loss.item()) 之类的代码. 这两种操作都会从 PyTorch 张量中取值, 并将其转换为 Python bool 或 float 类型. 这类代码在 eager 模式 PyTorch 中很快, 却会破坏所有跨迭代流水线; 对于一个 (尚未计算的) Lazy Tensor, 无法知道 torch.any() 应返回什么 (它控制代码将进入的分支) 或应打印哪些值. Lazy Tensors 完全看不到调用它的 Python 代码, 因此这种模式会迫使所有已累积的操作流水线 flush, 并使 CPU 捕获停顿, 等待 GPU 追上.

<span id="section-6-3"></span>

### 6.3 TorchInductor 加速比

<span id="table-03"></span>

![TorchDynamo 编译器后端的几何平均加速比与模型覆盖范围](./pytorch-2/table-03.png)

**表 3.** 不同 TorchDynamo 后端相对于 PyTorch eager 的几何平均加速比 (越高越好), 以及它们在每个基准测试套件中可工作的模型数量. 加速比计算只纳入可工作的模型. 比较时使用与 eager 模式相同的精度. N/A 表示后端不支持该配置. 本实验中, 所有后端都使用 TorchDynamo 作为捕获图的前端, 并接收相同的初始图. 其中包含 None, 用于估算不应用任何图优化时 TorchDynamo 的开销 (或加速比).

<span id="figure-04"></span>

![GPU 与 CPU 推理和训练中编译器加速比的累积分布](./pytorch-2/figure-04.png)

**图 4.** 相对于 PyTorch eager 模式的加速比累积分布函数 (CDF). 加速比 (x 轴) 越高越好, PyTorch eager 为 1×. 底层数据与 [表 3](#table-03) 相同.

[表 3](#table-03) 展示了在 3 个基准测试套件与多种配置中, TorchInductor 和另外 6 个 TorchDynamo 后端相对于 PyTorch eager 的几何平均加速比. 在本实验中, 我们保持图捕获机制 (TorchDynamo) 不变, 只改变后端编译器, 因此每个后端接收相同的输入图并产生相同的捕获开销. [图 4](#figure-04) 基于与 [表 3](#table-03) 相同的数据, 但合并 3 个基准测试套件, 展示加速比的累积分布函数 (CDF). 这有助于进一步了解加速比的分布方式.

在大多数情况下, TorchInductor 比其他后端更快. nvFuser [Sar22] 和 NNC [Zol21] 的加速比都集中在 1× 左右, 因为它们使用 eager PyTorch 内核, 只为 PyTorch 的一个子集生成代码. PyTorch/XLA [Pyt23a] 的性能变化更大, 许多情况下会产生很高的加速比, 另一些情况下则会大幅减速, 从而拉低平均值. 与其他配置相比, 它在 GPU float16 推理中表现更好, 尤其是在 TIMM 的视觉模型上. ONNX Runtime [Onn21], TVM [Che18e] 和 Hidet [Din23c] 仅支持推理, 并会因缺少算子实现等问题而无法运行许多模型. 在 CPU 上, ONNX Runtime 有 5 个模型的加速比超过 8× (TorchInductor 有 1 个), 但这些结果不能泛化, 超过一半的模型出现减速. 在 GPU 上, 除分别 4 个和 2 个模型外, TVM 与 Hidet 在所有模型上均产生减速. 在 CPU 上, TVM 对某些模型的表现明显更好, 对其他模型则大幅减速. 如果排除出现大幅减速的模型, TVM 会成为 TorchBench 上第二快的 CPU 推理后端 (仅次于 TorchInductor).

<span id="section-6-4"></span>

### 6.4 TorchInductor 加速比的来源

<span id="table-04"></span>

![TorchInductor 优化的消融研究](./pytorch-2/table-04.png)

**表 4.** 衡量从 TorchInductor 中移除优化所产生影响的消融研究. 测量 NVIDIA A100 GPU 上 float16 HuggingFace 相对于 eager PyTorch 的几何平均加速比. 括号中为与 "All TorchInductor optimizations" 的差值.

[表 4](#table-04) 逐一禁用优化, 并测量其对 HuggingFace 模型几何平均加速比的影响, 以此研究 TorchInductor 加速比的来源. 如果移除某项特定优化后减速更明显, 就说明它贡献了更多加速比.

TorchInductor 最大的加速来自将 pointwise, reduction 和 scatter 内核组合为数量更少的 fused kernel; 由于值可直接复用, 不必往返内存, 这减少了内存流量. 在 TorchInductor 中, 这些内核组合发生在两处: 1) 内联发生在 lowering 期间, 满足阈值时会将 pointwise kernel 的主体复制到它的所有 consumer 中. 2) 融合发生在调度期间, 会将剩余内核组合起来, 同时进行横向 consumer/consumer 融合. 这些 pass 有很多重叠, 因此我们还提供一个同时禁用二者的 "without fusion and inlining" 行. 同时缺少这两个 pass 时, TorchInductor 产生的是减速而非加速. 这是因为 TorchInductor 执行的分解会将较大的优化算子拆成许多更小的基础算子, 我们依靠融合重新组合它们, 以恢复 1× 性能.

[表 4](#table-04) 测量的其余优化包括: 1) 循环/布局重排使用投票算法重新排列内核中的循环, 并改变数据布局以匹配用法. 2) Matmul 模板为矩阵乘法使用带 pointwise epilogue fusion 的 Triton 模板, 而不是 cuBLAS/cuDNN. 自动调优器 (通过 mode="max-autotune" 启用) 会选择何时使用这些模板. 没有这项优化时, TorchInductor 完全不使用模板. 3) Parameter freezing 是一项仅用于推理的优化, 它通过常量折叠删除模型中只依赖参数的部分. 4) Pattern matching 使用图级 peephole 优化重写输入图, 然后再将其 lowering 到 TorchInductor. 5) Cudagraphs 可在 CUDA driver 层减少内核启动开销. 当静态分析表明该方法安全且配置中已启用时, TorchInductor 会自动使用它.

<span id="section-7"></span>

## 7 结论

本文提出了 PyTorch 的两个扩展 TorchDynamo 和 TorchInductor, 它们通过 PyTorch 程序中的图编译实现加速, 同时保留了 PyTorch eager 编程模型所具有的灵活性与易用性. 通过在 PyTorch 程序中启用图编译, 我们希望研究人员和从业者能够以更高的效率和灵活性解决规模更大, 更复杂的机器学习问题.


## 致谢

我们衷心感谢匿名审稿人和 shepherd Martin Maas, 他们的建议与反馈帮助改进了本文. 感谢 Brett Simmers 校对本文. 感谢所有参与 Triton 工作的人; 没有 Triton, TorchInductor 的 GPU 后端就不可能实现. 感谢 Intel PyTorch 团队: Guobing Chen, Leslie Fang, Jiong Gong, Xuan Liao, Yudong Si, Chuanqi Wang, Eikan Wang, Chunyuan Wu, Weiwen Xia, Xiaobing Zhang, Fan Zhao 和 Beilei Zheng. 他们的工作大幅改进了 TorchInductor 的 CPU 后端. 最后, 感谢数千名为 PyTorch 贡献代码的人. 正是无数贡献汇聚在一起, 成就了今天的 PyTorch, 没有这些贡献就不可能完成这项工作.

<span id="section-8"></span>

## 8 Artifact 附录

<span id="section-8-1"></span>

### 8.1 Artifact 摘要

这项工作的源代码包含在 PyTorch 中, 可从 [https://github.com/pytorch/pytorch/](https://github.com/pytorch/pytorch/) 获取. TorchDynamo 位于 torch/_dynamo 目录, TorchInductor 位于 torch/_inductor 目录. 用于复现论文结果的基准测试代码位于 [https://github.com/pytorch/pytorch/tree/main/benchmarks/dynamo](https://github.com/pytorch/pytorch/tree/main/benchmarks/dynamo).

本文包含大量实验, 全部运行完需要数周, 因此这里的说明将侧重于复现 TorchInductor 在 GPU 上的 HuggingFace 结果. 其他结果的复现工作流程与此非常相似, 并会在末尾说明. PyTorch 的 benchmarks/dynamo 目录中包含的 README.md 提供了更多说明.

<span id="section-8-2"></span>

### 8.2 Artifact 检查清单 (元信息)

- 二进制文件: 可从 [https://pytorch.org/](https://pytorch.org/) 获取发行版本
- 硬件: NVIDIA A100 GPU, Intel Xeon 8275CL CPU
- 指标: 相对于 PyTorch eager 模式的几何平均加速比
- 所需磁盘空间 (约): 50 GB
- 准备工作流程所需时间 (约): 1 小时
- 完成实验所需时间 (约): 对大多数实验而言, 每个后端, 每种配置 < 1 天
- 是否公开可用: 是
- 代码许可证 (如公开可用): BSD-3

<span id="section-8-3"></span>

### 8.3 说明

<span id="section-8-3-1"></span>

#### 8.3.1 如何访问.

- 源代码和基准测试代码: [https://github.com/pytorch/pytorch/](https://github.com/pytorch/pytorch/)
- PyTorch 二进制文件: [https://pytorch.org/](https://pytorch.org/)
- TorchBench: [https://github.com/pytorch/benchmark/](https://github.com/pytorch/benchmark/)

<span id="section-8-3-2"></span>

#### 8.3.2 硬件依赖.

- 要匹配本文配置: NVIDIA A100 GPU 和 Intel Xeon 8275CL CPU
- 基准测试可在配有 SM80+ 和 40GB+ 内存的 NVIDIA GPU 上运行, 大多数基准测试使用较少内存也可运行
- CPU 结果可以在没有 GPU 的情况下运行

<span id="section-8-3-3"></span>

#### 8.3.3 软件依赖.

- 较新的 Linux 发行版
- NVIDIA 内核驱动程序
- 与所选 PyTorch 版本兼容的 CUDA 版本
- 与所选 CUDA 兼容的 gcc/g++
- 已安装 Miniconda ([https://docs.conda.io/projects/miniconda/en/latest/](https://docs.conda.io/projects/miniconda/en/latest/))
- PyTorch (及其依赖项)
- 其他 Python 包: pandas, scipy, psutil 和 tqdm

<span id="section-8-4"></span>

### 8.4 安装

[https://pytorch.org/](https://pytorch.org/) 介绍了多种 PyTorch 安装方式. 可以使用以下命令完成包含依赖项的最小安装:

```shell
# create a new conda environment
conda create --name=pt2 python=3.10
conda activate pt2

# install dependencies for benchmark code
conda install pandas scipy psutil tqdm

# install PyTorch using release build
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 \
  -c pytorch -c nvidia
```

接下来, 下载 PyTorch 源代码以访问基准测试脚本:

```shell
# clone the PyTorch repository to get benchmark code
git clone --recursive --branch=release/2.1 \
  https://github.com/pytorch/pytorch

# benchmark code should be run from the root PyTorch directory
cd pytorch
```

<span id="section-8-5"></span>

### 8.5 实验工作流程

要复现 TorchInductor 在 GPU 上以 float16 运行 HuggingFace 推理时相对于 eager PyTorch 的加速比, 请运行:

```shell
TORCHINDUCTOR_MAX_AUTOTUNE=1 ./benchmarks/dynamo/huggingface.py \
  --performance --no-skip \
  -dcuda --float16 --inference \
  --inductor --freezing \
  --output=`pwd`/results.csv
```

该命令会下载 HuggingFace 模型, 分别在使用和不使用 TorchDynamo 的情况下运行模型, 并计算相对于 PyTorch eager 模式的加速比. 结果写入当前工作目录下的 results.csv. 如果运行其他实验, 应为每个实验将 `--output` 设置为唯一的绝对文件名.

<span id="section-8-6"></span>

### 8.6 评估与预期结果

所选输出文件 (results.csv) 应包含 46 个条目, 展示每个模型的加速比数值 (以及其他指标). 所有模型都应能够工作 (失败表示为零加速比), 所有加速比的几何平均值应与论文报告的加速比相近.

<span id="section-8-7"></span>

### 8.7 实验定制

上面的命令可以通过多种方式定制:

- 对于 3 个基准测试套件, 可将 `./benchmarks/dynamo/huggingface.py` 替换为脚本 `./benchmarks/dynamo/timm_models.py` 或 `./benchmarks/dynamo/torchbench.py`. 注意, TorchBench 需要执行额外安装步骤, 另两个套件会自动下载依赖项.
- 可以将 `-dcuda` 替换为 `-dcpu` 以使用 CPU
- 可以将 `--float16` 替换为 `--float32` 或 `--amp`
- 可以将 `--inference` 替换为 `--training`
- 可以将 `--inductor` 替换为 `--backend=eager` (表示 "None"), `--backend=nvfuser`, `--backend=nnc`, `--xla`, `--backend=onnxrt`, `--backend=tvm` 或 `--backend=hidet`.

注意, 每个后端都有不同的依赖项和设置说明.

- 可以移除 `--freezing` 和/或 `TORCHINDUCTOR_MAX_AUTOTUNE=1`, 以在 TorchInductor 中禁用这些优化. `torch/_inductor/config.py` 中还可以找到更多优化标志.
- 通过 `--help` 可以使用许多其他选项和后端.

本文的结果包含这些标志中大多数的笛卡尔积组合.

<span id="section-8-8"></span>

### 8.8 说明

- 加速比和模型覆盖结果与本文所示结果相比, 在新版本 PyTorch 中已有改善. 对于未来的比较, 我们建议运行最新版本的 PyTorch.
- 性能结果可能对硬件和 CUDA 版本等环境设置敏感, 因此预期会有一些小差异.
- TorchBench 和非 TorchInductor 后端需要执行额外安装步骤.
- 基于这些脚本的性能仪表板位于 [https://hud.pytorch.org/benchmark/compilers](https://hud.pytorch.org/benchmark/compilers).
