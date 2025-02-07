#!/bin/bash

# 数据
# /tcdata/round2_test_data.jsonl

# 模型路径
# /data/root/jupyter/modelscope/fintune/LLaMA-Factory/models/llama3.1_70b_lora_3bit_bf16_text

# 这里可以放入代码运行命令
echo "program start..."

# LD路径 py3.10.12
# export LD_LIBRARY_PATH=/usr/local/lib/python3.10.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# 启动vllm
echo "run_vllm start..."
bash run_vllm.sh > logs/run_vllm.log 2>&1 &
sleep 30

# 调用脚本生成数据
echo "run_llm_api start..."
bash run_llm_api.sh > logs/run_llm_api.log

# 生成最终answer文件
echo "run_base_merge start..."
bash run_base_merge.sh > logs/run_base_merge.log
# /app/results.jsonl

# python3 run.py