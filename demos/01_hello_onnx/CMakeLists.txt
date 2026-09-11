# Demo 1: Hello ONNX Runtime 构建脚本
#
# 依赖: 需要 ONNX Runtime 的 include 与 lib。
#   - 本项目已自带 onnxruntime1.26.0-win 的 C/C++ 头文件(位于仓库根目录)
#   - 编译时需要链接到你本机对应平台的 onnxruntime.lib / .so
#
# 使用方式(以 Windows + CMake 为例):
#   cmake -S . -B build -DONNXRUNTIME_LIB="C:/path/to/onnxruntime.lib"
#   cmake --build build --config Release
cmake_minimum_required(VERSION 3.19)
project(hello_onnx_demo CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 头文件路径: 指向本项目自带 onnxruntime 头文件目录
set(ONNXRUNTIME_INCLUDE_DIR
    "${CMAKE_SOURCE_DIR}/../../onnxruntime1.26.0-win")
# 库文件路径(由使用者通过 -D 指定),例如 onnxruntime.lib
if(NOT DEFINED ONNXRUNTIME_LIB)
    message(FATAL_ERROR "请通过 -DONNXRUNTIME_LIB=... 指定 onnxruntime 库文件路径")
endif()

add_executable(hello_onnx main.cpp)
target_include_directories(hello_onnx PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")
target_link_libraries(hello_onnx PRIVATE "${ONNXRUNTIME_LIB}")

# 运行前将 models 目录复制到可执行文件旁边,或直接使用相对路径运行
# 运行: ./hello_onnx   (请确保 models/linear.onnx 在当前工作目录下可见)