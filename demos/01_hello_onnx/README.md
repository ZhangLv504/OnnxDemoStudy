# Demo 1: Hello ONNX Runtime

第一个 demo，跑通 **加载模型 → 构造输入张量 → 推理 → 读取输出** 的最小闭环。

## 模型

`models/linear.onnx`（由 `scripts/gen_linear_onnx.py` 生成）

- 数学含义：`y = x @ W + b`
- 输入 `x`：shape `[1, 4]`，float32
- 权重 `W`：shape `[4, 3]`（initializer 固化在模型里）
- 偏置 `b`：shape `[3]`
- 输出 `y`：shape `[1, 3]`

## 代码脉络（main.cpp）

1. `Ort::Env` —— 运行时全局环境（日志级别、内存、线程池）。
2. `Ort::SessionOptions` + `Ort::Session` —— 配置并加载模型。
3. `Ort::MemoryInfo::CreateCpu` + `Ort::Value::CreateTensor` —— 构造输入张量。
4. `session.Run(...)` —— 执行推理，返回输出张量。
5. 读取输出 shape 与数据，打印。

## 预期输出

输入 `x = [1, 2, 3, 4]` 时，理论结果：

```
y[0] = 1·1 + 2·4 + 3·7 + 4·10 + 0.5 = 70.5
y[1] = 1·2 + 2·5 + 3·8 + 4·11 + 1.5 = 83.5
y[2] = 1·3 + 2·6 + 3·9 + 4·12 + 2.5 = 96.5
```

## 编译运行（Windows + CMake）

```bash
cmake -S . -B build -DONNXRUNTIME_LIB="C:/path/to/onnxruntime.lib"
cmake --build build --config Release
cd build
# 把 models 目录复制到 build 下，或直接以相对路径运行
./hello_onnx
```