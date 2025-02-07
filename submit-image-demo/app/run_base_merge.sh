./miniconda3/envs/vllm/bin/python3 ./src/merge_base.py \
--model llama3.1_70b_lora_3bit_bf16_text \
--stage test \
--input_file ./data/upload_merge.jsonl \
--base_file /tcdata/round2_test_data.jsonl
# --base_file ./data/round1_test_data_test.jsonl