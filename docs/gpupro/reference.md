---
title: "参考资料"
createTime: 2026/08/01 00:00:00
permalink: /gpupro/reference/
pageClass: gpupro-page
---

本书的主线内容位于第一至第四部分. 阅读过程中需要查询具体细节时, 可以使用下面的参考资料:

| 需要查询的内容 | 对应页面 |
|---|---|
| TIRx 语言特性的准确写法和语义 | **[TIRx 语言参考](/gpupro/tirx-language-reference/)** |
| 排查异步 GEMM 或 Flash Attention kernel 的卡死, 崩溃, 错误结果和性能下降 | **[调试 Warp-Specialized Kernels](/gpupro/debugging-warp-specialized-kernels/)** |
| 编译器内部机制与 lowering 流程 | **[编译器内部机制](/gpupro/compiler-internals/)** |

完整的 `tvm.tirx` Python API 请参阅
[TVM 官方文档](https://tvm.apache.org/docs/).

第二部分介绍了 TIRx 编程模型 ([TIRx 入门](/gpupro/tirx-introduction/)) 和 tensor layout 模型
([TIRx Layout API](/gpupro/tirx-layout-api/)).
