#!/bin/bash

# 检查命令行参数
if [ $# -eq 0 ]; then
    echo "用法: $0 <源文件.cu> [输出文件名]"
    echo "示例: $0 mma.cu"
    echo "示例: $0 mma.cu my_program"
    exit 1
fi

# 获取输入文件名
SOURCE_FILE="$1"

# 检查源文件是否存在
if [ ! -f "$SOURCE_FILE" ]; then
    echo "错误: 文件 '$SOURCE_FILE' 不存在"
    exit 1
fi

# 获取输出文件名（如果未提供，则使用源文件名去掉扩展名）
if [ $# -ge 2 ]; then
    OUTPUT_FILE="$2"
else
    # 从源文件名中提取基本名称（去掉路径和扩展名）
    BASENAME=$(basename "$SOURCE_FILE" .cu)
    OUTPUT_FILE="${BASENAME}.out"
fi

echo "编译文件: $SOURCE_FILE"
echo "输出文件: $OUTPUT_FILE"

# 编译并运行
nvcc -I/data/solution-sdk/jiaao1/kernels-workspace/kernels/3rdparty/cutlass/include \
    --std=c++17 \
    -o "$OUTPUT_FILE" \
    "$SOURCE_FILE" \
&& echo "编译成功，正在运行..." \
&& "./$OUTPUT_FILE" \
&& echo "程序执行完成，清理临时文件..." \
&& rm "$OUTPUT_FILE"