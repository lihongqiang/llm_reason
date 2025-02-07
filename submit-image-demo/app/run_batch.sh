CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# VLLM_ATTENTION_BACKEND=FLASHINFER \
./miniconda3/envs/vllm/bin/python3 src/batch_vllm.py \
--model Reflection-Llama-3.1-8B \
--answer_name Reflection-Llama-3.1-8B \
--gpu_memory_utilization 0.9 \
--tp 8 \
--max_model_len 8192 \
--check_model \
--input_file ./data/round1_train_data_fix.jsonl \


# Infinity-Instruct-7M-Gen-Llama3_1-8B 999/1415
# Infinity-Instruct-7M-Gen-mistral-7B 951/1415
# Meta-Llama-3.1-8B-Instruct 992/1415
# Meta-Llama-3.1-8B-Instruct+relfection 985/1415

# glm-4-9b-chat 1032/1415
# gemma-2-9b-it 1068/1415
# qwen2 1041/1415