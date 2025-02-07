CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 src/batch_vllm_gen_data.py \
--model DeepSeek-Coder-V2-Instruct \
--input_file ./data/round1_train_data_fix.jsonl \
--gpu_memory_utilization 0.98 \
--retry_total 20 \
--same_break \
--max_model_len 8192 \
--tp 8 \
--reflection \
--temperature 0.7 \
--top_p 0.95 \
--build_answer 

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 src/batch_vllm_gen_data.py \
--model Meta-Llama-3.1-405B-Instruct-FP8 \
--input_file ./data/round1_train_data_fix.jsonl \
--gpu_memory_utilization 0.98 \
--retry_total 20 \
--same_break \
--max_model_len 8192 \
--tp 8 \
--reflection \
--temperature 0.9 \
--top_p 0.9 \
--build_answer 