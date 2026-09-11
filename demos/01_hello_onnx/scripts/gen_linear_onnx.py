"""
生成 linear.onnx 示例模型（仅用于 Demo 1 学习演示）。

数学含义: y = x @ W + b
- 输入 x:  shape [1, 4]
- 权重 W:  initializer, shape [4, 3]
- 偏置 b:  initializer, shape [3]
- 输出 y:  shape [1, 3]

依赖: pip install onnx numpy
运行: python gen_linear_onnx.py
"""
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def main():
    # 1. 定义 图输入 / 图输出
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])

    # 2. 定义常量权重 W[4,3] 和偏置 b[3] (作为 initializer 固化进模型)
    W_np = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ],
        dtype=np.float32,
    )
    b_np = np.array([0.5, 1.5, 2.5], dtype=np.float32)

    W = numpy_helper.from_array(W_np, name="W")
    b = numpy_helper.from_array(b_np, name="b")

    # 3. 算子节点
    matmul_node = helper.make_node(
        "MatMul", inputs=["x", "W"], outputs=["matmul_out"], name="MatMul_W"
    )
    add_node = helper.make_node(
        "Add", inputs=["matmul_out", "b"], outputs=["y"], name="Add_b"
    )

    # 4. 组装图与模型
    graph = helper.make_graph(
        nodes=[matmul_node, add_node],
        name="linear_demo",
        inputs=[x],
        outputs=[y],
        initializer=[W, b],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = onnx.checker.check_model(model) or onnx.IR_VERSION
    onnx.checker.check_model(model)

    # 4. 保存
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "..", "models")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "linear.onnx")
    onnx.save(model, out_path)
    print(f"[ok] 已生成模型: {out_path}")

    # 6. 打印模型结构摘要
    print("\n=== 模型结构摘要 ===")
    print(f"opset : {model.opset_import[0].version}")
    print("输入  :", [(i.name, i.type.tensor_type.shape) for i in model.graph.input])
    print("初始权重:", [(t.name, [d for d in t.dims]) for t in model.graph.initializer])
    print("算子节点:", [(n.op_type, n.input, n.output) for n in model.graph.node])


if __name__ == "__main__":
    main()