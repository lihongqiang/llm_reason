./miniconda3/envs/vllm/bin/python3 ./src/base_llm_api_args.py \
--model gemma2_lora_8bit_bf16_text \
--stage test \
--try_true \
--input_file ./data/round1_test_data_test.jsonl \
--port 8000
