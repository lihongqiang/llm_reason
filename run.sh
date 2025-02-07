#######################
# 生成训练数据
#######################
# 部署base模型vllm服务
CUDA_VISIBLE_DEVICES=0 \   
python -m vllm.entrypoints.openai.api_server  \  
--model LLM-Research/Meta-Llama-3.1-70B-Instruct  \  
--served-model-name Meta-Llama-3.1-70B-Instruct \  
--max-model-len=8192 \  
--trust-remote-code \  
--enforce-eager \  
--gpu_memory_utilization=1 \  
--tensor-parallel-size=1 \  
--dtype=bfloat16 \  
--max-log-len=10 \  
--disable-log-stats \  
--uvicorn-log-level=warning \  
--max_num_seqs=2048 \  
--port 8000 &

# 调用api接口生成初始答案
# generate upload.jsonl upload_merge.jsonl gen_dpo.jsonl error.jsonl
python3 ./src/base_llm_api_args.py \
--model Meta-Llama-3.1-70B-Instruct \
--stage gen_dpo \
--mode gen_dpo \
--input_file ./data/round1_train_data_fix.jsonl \
--eval \
--port 8000

# 合并题目的多个问题，多次尝试确保生成正确的答案
# generate gen_dpo_true.jsonl
python3 ./src/base_llm_api_args.py \
--model Meta-Llama-3.1-70B-Instruct \
--stage gen_dpo \
--mode gen_dpo \
--eval \
--input_file ./data/upload_merge_base.jsonl \
--port 8000 \
--try_true

# 对gen_dpo_true.jsonl首尾增加中括号[]，得到最终模型SFT训练数据round1_train_llama3.1_sft_self.jsonl
train_data_path=./data/round1_train_llama3.1_sft_self_test.jsonl
rm $train_data_path
touch $train_data_path
echo '[' > $train_data_path
cat ./data/gen_dpo_true.jsonl >> $train_data_path
echo ']' >> $train_data_path

# 增加训练数据文件到LLaMa-Factory/data/dataset_info.json
sed -i '1a "round1_train_llama3.1_sft_self\":{\"file_name\": \"./data/round1_train_llama3.1_sft_self_test.jsonl\"},' ./src/LLaMA-Factory/data/dataset_info.json


#######################
# 开始模型SFT训练
#######################

CUDA_LAUNCH_BLOCKING=1 \
TF_ENABLE_ONEDNN_OPTS=0 \
OMP_NUM_THREADS=16 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
llamafactory-cli train examples/train_lora/llama3.1-70b_self.yaml


#######################
# 模型合并
#######################
llamafactory-cli export examples/merge_lora/llama3.1_lora_sft.yaml


#######################
# 模型量化压缩
#######################
# 生成量化校验数据集
# generate data/round1_text.json
python3 src/build_quant_val_data.py

# 开始模型量化压缩
CUDA_VISIBLE_DEVICES=0,1,2 \
TF_ENABLE_ONEDNN_OPTS=0 \
llamafactory-cli export examples/merge_lora/llama3.1_quant_bf16.yaml


#######################
# 量化模型推理
#######################
# 部署sft模型vllm api服务，这里采用A100*8，gpu_memory_utilization设置成0.4，占用显存32G
CUDA_VISIBLE_DEVICES=0 \
python -m vllm.entrypoints.openai.api_server  \
--model ./models/llama3.1_70b_lora_3bit_bf16_text  \
--served-model-name llama3.1_70b_lora_3bit_bf16_text \
--max-model-len=1024 \
--dtype=float16 \
--kv-cache-dtype=fp8 \
--block-size=16 \
--trust-remote-code \
--enforce-eager \
--gpu_memory_utilization=0.4 \
--tensor-parallel-size=1 \
--max-log-len=10 \
--disable-log-stats \
--uvicorn-log-level=warning \
--max-num-seqs=256 \
--port 8000

# 调用接口生成答案
python3 ./src/base_llm_api_args.py \
--model llama3.1_70b_lora_3bit_bf16_text \
--stage test \
--mode infer \
--input_file ./data/round1_test_data.jsonl \
--port 8000 

#### 合并base文件生成answer文件
python3 ./src/test_merge_base.py \
--model Meta-Llama-3.1-70B-Instruct \
--stage test \
--input_file ./data/upload_merge.jsonl \
--base_file ./data/round1_test_data.jsonl 