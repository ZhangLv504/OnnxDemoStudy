# 知识点 Demo 03：算子注册与 Kernel 执行机制

- **关联知识点**：算子（Operator）↔ 可执行实现（Kernel）↔ 图的调度
- **核心问题**：图上那个名为 `Conv` 的节点，ORT 是怎么知道"该跑哪段 C++ 代码"的？

---

## 1. 关键区分：Schema（算子规范）vs Kernel（可执行实现）

在 ONNX Runtime 的世界里，**同一个算子被拆成两个层面**：

| 层面 | 是什么 | 回答的问题 | 示例 |
|------|--------|-----------|------|
| **Op Schema** | 算子的"形状/类型约束"定义 | 这个算子**接受什么输入、输出什么、有哪些属性** | `Conv` 至少 2 个输入、输出 1 个；输入必须是张量 |
| **Kernel** | 算子的"具体实现" | 真拿到数据时**怎么算** | `Conv` 的 CPU 卷积循环 / CUDA 的 cuDNN 实现 |

**区分它俩对理解推理极其重要**：
- `Conv`（概念/约束）只有一个；
- `Conv` 的 Kernel 却可以有**很多个**——CPU 一个、CUDA 一个、TensorRT 一个，行为可能略有不同但"输入输出契约"一致。

## 2. 从「节点」到「Kernel」的映射链

一个图节点携带的信息，正是用来"找到正确 Kernel"的**钥匙**：

```
图节点 (Node)
  ├─ op_type      : "Conv"
  ├─ domain       : "ai.onnx"        ← 算子所属命名空间
  ├─ since_version: 11                ← 算子版本（决定 schema 兼容）
  ├─ 输入类型     : float32[1,3,224,224]   ← 类型/形状
  └─ 所在 EP      : CPU_CUDA          ← 在哪执行
        │
        ▼  查「注册表 KernelRegistry」
   命中一个 KernelDef（算子+EP+版本+类型约束都匹配）
        │
        ▼  Session 建立时实例化
   Kernel 实例（持有权重等预设状态）
        │
        ▼  Run 时调用
   Kernel::Compute / KernelCreateFunc 执行 → 产出输出张量
```

**两条主线，再次呼应 Demo 01**：
- **建 Session**：为每个节点查注册表、**绑定/实例化 Kernel**；
- **Run**：依次调用每个 Kernel 的 Compute。

## 3. Kernel 靠什么被"匹配"？—— KernelDef

从 [onnxruntime_cxx_api.h](file:///workspace/onnxruntime1.26.0-win/onnxruntime_cxx_api.h) 的 `KernelDefBuilder` 可以看到，一份 Kernel 定义（`KernelDef`）要声明这几类"匹配条件"：

```cpp
KernelDefBuilder builder;
builder.SetOperatorType("Conv");          // 匹配 op_type
builder.SetDomain("ai.onnx");             // 匹配 domain
builder.SetSinceVersion(1, 15);           // 匹配算子版本区间
builder.SetExecutionProvider("CPU");      // 匹配哪个 EP 执行
builder.AddTypeConstraint("T", {          // 类型约束：允许哪些数据类型
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16});
KernelDef def = builder.Build();
```

```cpp
// 再把 KernelDef + 一个"创建函数"放进注册表（逻辑示意）
KernelRegistry registry;
registry.AddKernel(&def, /* kernel_create_func = */ [](...){ return new ConvKernel(...); }, nullptr);
```

> 一句话：**KernelDef 描述"我要处理哪种算子、哪个 domain、哪个版本段、哪个 EP、哪些数据类型"；KernelRegistry 把这些定义聚合起来，供建图时按节点信息查找。** 这就是上一节"映射链"里"查注册表"的实际载体。

## 4. 一个算子，多份 Kernel——为什么？

因为 `KernelDef` 里带上了 **`SetExecutionProvider("CPU"/"CUDA")`** 这个字段。同一个 `Conv` 节点：

- 若会话启用了 CUDA EP → 命中 `Conv` 的 CUDA Kernel（内部调 cuDNN，跑在 GPU）；
- 否则 → 命中 CPU Kernel（普通循环，跑在 CPU）。

这正是 **Demo 04（EP 切换）** 的底层机制：**EP 的切换，本质是"同一个算子替换成绑定到不同 KernelDef 的 Kernel"。**

## 5. 自定义算子的位置

Schema 与 Kernel 分离的设计，让"新增算子"变得简单：新算子 = **写一个 Schema（让 ORT 认识它）+ 写一个/多个 Kernel（让 ORT 能执行它）**。这正是后续 **Demo 07（自定义算子 Custom Op）** 会落地的内容。

> 顺带一提：本仓库 GPU 版里的 [cuda_context.h](../onnxruntime1.26.0-win-gpu/core/providers/cuda/cuda_context.h) 就是给「CUDA 自定义算子」用的——在自定义 Kernel 的 Compute 里，通过 `CudaContext` 去复用 ORT 已经建好的 cuda stream / cuDNN / cuBLAS 句柄，而不是自己再建一套。

## 6. 推理流程视角的一句话总结

> ORT 的推理 = **按图拓扑遍历节点 → 每个节点用(KernelDef 匹配到的)Kernel 实例执行 Compute**。算子(Schema)定义"长什么样"，Kernel 决定"怎么算"，EP 决定"在哪算"。

## 7. 自测题（掌握本知识点）

1. Schema 和 Kernel 的区别是什么？为什么同一个算子可以有多个 Kernel？
2. `KernelDefBuilder` 里哪个字段决定"CUDA 的 Conv 和 CPU 的 Conv"是两份不同的注册？
3. 图中节点要用哪些信息去注册表里"找 Kernel"？（提示：至少列 3 个）
4. 为什么把 `KernelDef` 和"KernelCreateFunc"分开、而不是直接把执行代码塞进 kernel def？（提示：与"建 Session 时不真正计算"有什么关系）

---

> 下一个知识点：**Demo 04 — Execution Provider（CPU/CUDA）**，把"算子在哪执行"落地为 API 用法。