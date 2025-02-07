CUDA_VISIBLE_DEVICES=4,5 \
./miniconda3/envs/vllm/bin/python3 src/batch_build_datasets_vllm.py \
--model Meta-Llama-3.1-70B-Instruct \
--answer_name Meta-Llama-3.1-70B-Instruct \
--retry_total 10 \
--gpu_memory_utilization 0.95 \
--tp 2 \
--reflection \
--input_file ./data/round1_train_data_fix.jsonl 

CUDA_VISIBLE_DEVICES=2,3 \
./miniconda3/envs/vllm/bin/python3 src/batch_build_datasets_vllm.py \
--model Qwen2-72B-Instruct \
--answer_name Qwen2-72B-Instruct \
--retry_total 10 \
--gpu_memory_utilization 0.95 \
--tp 2 \
--reflection \
--input_file ./data/round1_train_data_fix.jsonl


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
./miniconda3/envs/vllm/bin/python3 src/batch_build_datasets_vllm.py \
--model Meta-Llama-3.1-70B-Instruct \
--answer_name Meta-Llama-3.1-70B-Instruct \
--out_tag 'LogiQuest' \
--retry_total 10 \
--gpu_memory_utilization 0.95 \
--tp 8 \
--reflection \
--input_file datasets/LogiQuest_25000.jsonl 

# llama3.1-70b 1231/1415=0.8699
# qwen2-72b 1182/1415

# llama3.1-70b top1w 27955/40234