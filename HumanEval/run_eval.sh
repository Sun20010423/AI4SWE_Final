#!/bin/bash

# 设置环境变量
export LANG=python
export OUTPUT_DIR=output
export MODEL_PATH="../finetune/models/deepseek-coder-1.3b-instruct"  # 使用绝对路径
export CUDA_VISIBLE_DEVICES=0

# 如果输出目录不存在，则创建它
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# 构建命令和参数
COMMAND=("python" "eval_instruct.py"
         "--model" "$MODEL_PATH"  # 使用绝对路径
         "--output_path" "${OUTPUT_DIR}/${LANG}.deepseek-coder-1.3b-instruct.jsonl"
         "--language" "$LANG"
         "--temp_dir" "$OUTPUT_DIR")

echo "即将执行的命令: ${COMMAND[*]}"

# 执行命令
"${COMMAND[@]}"
if [ $? -eq 0 ]; then
    echo "命令成功执行"
else
    echo "执行命令时出错"
fi