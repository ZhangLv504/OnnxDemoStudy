// ============================================================================
// Demo 1: Hello ONNX Runtime —— 最小可运行闭环
//
// 步骤:
//   1. 创建 Ort::Env（运行时全局环境，管理线程池/日志/内存）
//   2. 构造 Ort::SessionOptions，加载 ONNX 模型为 Ort::Session
//   3. 分配输入内存，构造输入张量 Ort::Value
//   4. 调用 session.Run(输入名, 输入张量, 输出名) 拿到输出，打印结果
//
// 模型: linear.onnx  (y = x @ W + b)
//   x shape [1,4] ; W shape [4,3] ; b shape [3] ; y shape [1,3]
// ============================================================================
#include <iostream>
#include <vector>
#include <string>

#include "onnxruntime_cxx_api.h"  // C++ 封装头 (在同一工程 include 目录内)

int main() {
    try {
        // ------------------------------------------------------------------
        // (1) 创建环境。第一个参数: 日志级别; 第二个参数: 环境名(约)用于调试。
        //     注意: 一个进程中通常只需创建一次 Env (作为全局/单例即可)。
        // ------------------------------------------------------------------
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "hello_onnx_demo");

        // ------------------------------------------------------------------
        // (2) 初始化 SessionOptions 并加载模型。
        //     模型文件路径可改为绝对路径或通过参数传入。
        // ------------------------------------------------------------------
        const std::string model_path = "models/linear.onnx";
        Ort::SessionOptions session_options;
        // 设置图优化级别(默认即为 ALL);学习阶段先显式写上,方便后续 demo 对比。
        session_options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);

        Ort::Session session(env, model_path.c_str(), session_options);

        // ------------------------------------------------------------------
        // (3) 准备输入数据与输入张量。
        //     linear.onnx 输入 x 的 shape = [1, 4], 类型 float。
        // ------------------------------------------------------------------
        std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<int64_t> input_shape = {1, 4};

        // 内存信息唯一标识符: CPU 上、无 TENSOR 分配器约束(用默认 allocator)。
        Ort::MemoryInfo mem_info =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        // CreateTensor(info, 数据首地址, 元素个数, shape, 元素类型)
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            mem_info, input_data.data(), input_data.size(), input_shape.data(),
            input_shape.size());

        // ------------------------------------------------------------------
        // (4) 执行推理。
        //     Run(run_name(可为nullptr), input_names, input_tensors, count,
        //          output_names, output_count)
        // ------------------------------------------------------------------
        const char* input_names[] = {"x"};
        const char* output_names[] = {"y"};
        auto output_tensors =
            session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1,
                        output_names, 1);

        // 解析输出: 取第 0 个输出, 读 shape 与数据。
        auto& output = output_tensors[0];
        auto type_info = output.GetTensorTypeAndShapeInfo();
        auto shape = type_info.GetShape();
        size_t count = type_info.GetElementCount();
        float* out_ptr = output.GetTensorMutableData<float>();  // 取数据首地址
        std::vector<float> out_vals(out_ptr, out_ptr + count);

        // ------------------------------------------------------------------
        // 打印结果
        // ------------------------------------------------------------------
        std::cout << "=== 推理结果 ===\n";
        std::cout << "输出 y shape = [";
        for (size_t i = 0; i < shape.size(); ++i) {
            std::cout << shape[i] << (i + 1 < shape.size() ? ", " : "]\n");
        }
        for (size_t i = 0; i < count; ++i) {
            std::cout << "y[" << i << "] = " << out_vals[i] << "\n";
        }

        // 预期数学结果: y = x@W + b
        //   y[0] = 1*1+2*4+3*7+4*10   + 0.5 = 70.5
        //   y[1] = 1*2+2*5+3*8+4*11   + 1.5 = 83.5
        //   y[2] = 1*3+2*6+3*9+4*12   + 2.5 = 96.5
        return 0;
    } catch (const Ort::Exception& e) {
        // ORT 的异常基类,可取出具体错误信息。
        std::cerr << "ONNX Runtime 异常: " << e.what() << "\n";
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "标准异常: " << e.what() << "\n";
        return -1;
    }
}