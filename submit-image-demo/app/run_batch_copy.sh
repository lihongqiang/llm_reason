CUDA_VISIBLE_DEVICES=4,5,6,7 \
python3 src/batch_vllm.py \
--model Infinity-Instruct-7M-Gen-mistral-7B \
--answer_name Infinity-Instruct-7M-Gen-mistral-7B \
--gpu_memory_utilization 0.9 \
--tp 4 \
--max_model_len 4096 \
--check_model \
--input_file ./data/round1_train_data_fix.jsonl