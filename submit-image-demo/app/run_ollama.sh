python3 src/batch_ollama.py \
--model Qwen2-7B-Instruct \
--answer_name Qwen2-7B-Instruct \
--input_file ./data/round1_train_data_fix.jsonl \
--api_type discover \
--check_model 

# python3 src/batch_ollama.py \
# --model Reflection-Llama-3.1-70B.Q2_K_L \
# --answer_name Reflection-Llama-3.1-70B.Q2_K_L \
# --input_file ./data/round1_train_data_fix.jsonl \
# --api_type ollama \
# --port 11434 \
# --check_model 