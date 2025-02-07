# CUDA_VISIBLE_DEVICES=0 \
# ./miniconda3/envs/vllm/bin/python3 src/batch_vllm_retry.py \
# --model glm-4-9b-chat  \
# --lora glm-4-9b-chat \
# --answer_name glm-4-9b-chat  \
# --gpu_memory_utilization 0.9 \
# --tp 1 \
# --retry_total 1 \
# --max_model_len 8192 \
# --check_model \
# --input_file ./data/round1_train_data_fix.jsonl  

# CUDA_VISIBLE_DEVICES=1 \
# ./miniconda3/envs/vllm/bin/python3 src/batch_vllm_retry.py \
# --model Qwen2-7B-Instruct  \
# --lora Qwen2-7B-Instruct \
# --answer_name Qwen2-7B-Instruct  \
# --gpu_memory_utilization 0.9 \
# --tp 1 \
# --retry_total 1 \
# --max_model_len 8192 \
# --check_model \
# --input_file ./data/round1_train_data_fix.jsonl  

# CUDA_VISIBLE_DEVICES=1 \
# ./miniconda3/envs/vllm/bin/python3 src/batch_vllm_retry.py \
# --model Meta-Llama-3.1-8B-Instruct  \
# --lora Meta-Llama-3.1-8B-Instruct \
# --answer_name Meta-Llama-3.1-8B-Instruct  \
# --gpu_memory_utilization 0.9 \
# --tp 1 \
# --retry_total 1 \
# --max_model_len 8192 \
# --check_model \
# --input_file ./data/round1_train_data_fix.jsonl  

# glm4 base try 5
CUDA_VISIBLE_DEVICES=0 \
./miniconda3/envs/vllm/bin/python3 src/batch_vllm_retry.py \
--model glm-4-9b-chat  \
--answer_name merge_model  \
--gpu_memory_utilization 0.95 \
--tp 1 \
--retry_total 5 \
--max_model_len 8192 \
--check_model \
--input_file ./data/round1_train_data_fix.jsonl  

# glm4 sft try 5
CUDA_VISIBLE_DEVICES=1 \
./miniconda3/envs/vllm/bin/python3 src/batch_vllm_retry.py \
--model glm-4-9b-chat  \
--lora glm-4-9b-chat \
--answer_name merge_model  \
--gpu_memory_utilization 0.95 \
--tp 1 \
--retry_total 5 \
--max_model_len 8192 \
--check_model \
--input_file ./data/round1_train_data_fix.jsonl  

# glm4 reflection try 5
CUDA_VISIBLE_DEVICES=2 \
./miniconda3/envs/vllm/bin/python3 src/batch_vllm_retry.py \
--model glm-4-9b-chat  \
--answer_name merge_model  \
--gpu_memory_utilization 0.95 \
--tp 1 \
--retry_total 5 \
--max_model_len 8192 \
--check_model \
--reflection \
--input_file ./data/round1_train_data_fix.jsonl  

# glm4 sft+reflection try 5
CUDA_VISIBLE_DEVICES=3 \
./miniconda3/envs/vllm/bin/python3 src/batch_vllm_retry.py \
--model glm-4-9b-chat  \
--lora glm-4-9b-chat \
--answer_name merge_model  \
--gpu_memory_utilization 0.95 \
--tp 1 \
--retry_total 5 \
--max_model_len 8192 \
--check_model \
--reflection \
--input_file ./data/round1_train_data_fix.jsonl 


# llama3.1 sft+reflection try 5
CUDA_VISIBLE_DEVICES=7 \
./miniconda3/envs/vllm/bin/python3 src/batch_vllm_retry.py \
--model Meta-Llama-3.1-8B-Instruct  \
--lora Meta-Llama-3.1-8B-Instruct \
--answer_name merge_model  \
--gpu_memory_utilization 0.95 \
--tp 1 \
--retry_total 5 \
--max_model_len 8192 \
--check_model \
--reflection \
--input_file ./data/round1_train_data_fix.jsonl 


CUDA_VISIBLE_DEVICES=0 \
./miniconda3/envs/vllm/bin/python3 src/batch_vllm_retry.py \
--model Meta-Llama-3.1-8B-Instruct,glm-4-9b-chat  \
--lora Meta-Llama-3.1-8B-Instruct,glm-4-9b-chat \
--answer_name merge_model  \
--gpu_memory_utilization 0.95 \
--tp 1 \
--retry_total 5 \
--max_model_len 8192 \
--check_model \
--reflection \
--input_file ./data/round1_train_data_fix.jsonl  


CUDA_VISIBLE_DEVICES=1,6 \
./miniconda3/envs/vllm/bin/python3 src/batch_vllm_retry.py \
--model Meta-Llama-3.1-8B-Instruct,glm-4-9b-chat,Qwen2-7B-Instruct  \
--lora Meta-Llama-3.1-8B-Instruct,glm-4-9b-chat,Qwen2-7B-Instruct \
--answer_name merge_model  \
--gpu_memory_utilization 0.95 \
--tp 2 \
--retry_total 5 \
--max_model_len 8192 \
--check_model \
--input_file ./data/round1_train_data_fix.jsonl  


CUDA_VISIBLE_DEVICES=0 \
./miniconda3/envs/vllm/bin/python3 src/batch_vllm_retry.py \
--model Meta-Llama-3.1-8B-Instruct \
--lora Meta-Llama-3.1-8B-Instruct \
--lora_dir lora_reflection \
--answer_name Meta-Llama-3.1-8B-Instruct  \
--gpu_memory_utilization 0.95 \
--tp 1 \
--retry_total 5 \
--max_model_len 8192 \
--check_model \
--reflection \
--input_file ./data/round1_train_data_fix.jsonl 

CUDA_VISIBLE_DEVICES=1 \
./miniconda3/envs/vllm/bin/python3 src/batch_vllm_retry.py \
--model glm-4-9b-chat \
--lora glm-4-9b-chat \
--lora_dir lora_reflection \
--answer_name glm-4-9b-chat  \
--gpu_memory_utilization 0.95 \
--tp 1 \
--retry_total 5 \
--max_model_len 8192 \
--check_model \
--reflection \
--input_file ./data/round1_train_data_fix.jsonl 

CUDA_VISIBLE_DEVICES=2 \
./miniconda3/envs/vllm/bin/python3 src/batch_vllm_retry.py \
--model Qwen2-7B-Instruct \
--lora Qwen2-7B-Instruct \
--lora_dir lora_reflection \
--answer_name Qwen2-7B-Instruct  \
--gpu_memory_utilization 0.95 \
--tp 1 \
--retry_total 5 \
--max_model_len 8192 \
--check_model \
--reflection \
--input_file ./data/round1_train_data_fix.jsonl 

CUDA_VISIBLE_DEVICES=3 \
./miniconda3/envs/vllm/bin/python3 src/batch_vllm_retry.py \
--model Meta-Llama-3.1-8B-Instruct  \
--model_dir models_merge_reflection \
--answer_name Meta-Llama-3.1-8B-Instruct   \
--gpu_memory_utilization 0.95 \
--tp 1 \
--retry_total 5 \
--max_model_len 8192 \
--check_model \
--reflection \
--input_file ./data/round1_train_data_fix.jsonl \
--temperature 1.0



CUDA_VISIBLE_DEVICES=2 \
./miniconda3/envs/vllm/bin/python3 src/batch_vllm_retry.py \
--model glm-4-9b-chat  \
--model_dir models_merge_reflection_big \
--answer_name glm-4-9b-chat   \
--gpu_memory_utilization 0.95 \
--tp 1 \
--retry_total 5 \
--max_model_len 8192 \
--check_model \
--reflection \
--input_file ./data/round1_train_data_fix.jsonl \
--temperature 0.8 \
--top_p 0.8

CUDA_VISIBLE_DEVICES=3 \
./miniconda3/envs/vllm/bin/python3 src/batch_vllm_retry.py \
--model Meta-Llama-3.1-8B-Instruct  \
--model_dir models_merge_reflection_big \
--answer_name Meta-Llama-3.1-8B-Instruct   \
--gpu_memory_utilization 0.95 \
--tp 1 \
--retry_total 5 \
--max_model_len 8192 \
--check_model \
--reflection \
--input_file ./data/round1_train_data_fix.jsonl \
--temperature 0.6 \
--top_p 0.9

CUDA_VISIBLE_DEVICES=2 \
./miniconda3/envs/vllm/bin/python3 src/batch_vllm_retry.py \
--model Qwen2-7B-Instruct  \
--model_dir models_merge_reflection_big \
--answer_name Qwen2-7B-Instruct   \
--gpu_memory_utilization 0.95 \
--tp 1 \
--retry_total 5 \
--max_model_len 8192 \
--check_model \
--reflection \
--input_file ./data/round1_train_data_fix.jsonl


CUDA_VISIBLE_DEVICES=6 \
./miniconda3/envs/vllm/bin/python3 src/batch_vllm_retry.py \
--model Meta-Llama-3.1-8B-Instruct  \
--model_dir models_merge_reflection_big \
--answer_name Meta-Llama-3.1-8B-Instruct   \
--gpu_memory_utilization 0.95 \
--tp 1 \
--retry_total 5 \
--max_model_len 8192 \
--check_model \
--reflection \
--input_file ./data/round1_train_data_fix.jsonl \
--temperature 0.8


CUDA_VISIBLE_DEVICES=5 \
./miniconda3/envs/vllm/bin/python3 src/batch_vllm_retry.py \
--model Qwen2-7B-Instruct  \
--model_dir models_merge_reflection_big \
--answer_name Qwen2-7B-Instruct   \
--gpu_memory_utilization 0.95 \
--tp 1 \
--retry_total 5 \
--max_model_len 8192 \
--check_model \
--reflection \
--input_file ./data/round1_train_data_fix.jsonl \
--temperature 0.6

CUDA_VISIBLE_DEVICES=6 \
./miniconda3/envs/vllm/bin/python3 src/batch_vllm_retry.py \
--model Meta-Llama-3.1-8B-Instruct  \
--model_dir models_merge_reflection_big \
--answer_name Meta-Llama-3.1-8B-Instruct   \
--gpu_memory_utilization 0.95 \
--tp 1 \
--retry_total 5 \
--max_model_len 8192 \
--check_model \
--reflection \
--input_file ./data/round1_train_data_fix.jsonl \
--temperature 0.8

CUDA_VISIBLE_DEVICES=7 \
./miniconda3/envs/vllm/bin/python3 src/batch_vllm_retry.py \
--model Qwen2-7B-Instruct  \
--model_dir models_merge_reflection_big \
--answer_name Qwen2-7B-Instruct   \
--gpu_memory_utilization 0.95 \
--tp 1 \
--retry_total 5 \
--max_model_len 8192 \
--check_model \
--reflection \
--input_file ./data/round1_train_data_fix.jsonl \
--temperature 0.8







CUDA_VISIBLE_DEVICES=6 \
./miniconda3/envs/vllm/bin/python3 src/batch_vllm_retry.py \
--model Meta-Llama-3.1-8B-Instruct,glm-4-9b-chat,Qwen2-7B-Instruct  \
--lora_dir lora_reflection \
--lora Meta-Llama-3.1-8B-Instruct,glm-4-9b-chat,Qwen2-7B-Instruct \
--answer_name merge_model  \
--gpu_memory_utilization 0.95 \
--tp 1 \
--retry_total 5 \
--max_model_len 8192 \
--check_model \
--reflection \
--input_file ./data/round1_train_data_fix.jsonl \


CUDA_VISIBLE_DEVICES=0 \
./miniconda3/envs/vllm/bin/python3 src/batch_vllm_retry.py \
--model Meta-Llama-3.1-8B-Instruct  \
--model_dir models_merge \
--answer_name Meta-Llama-3.1-8B-Instruct   \
--gpu_memory_utilization 0.95 \
--tp 1 \
--retry_total 5 \
--max_model_len 8192 \
--check_model \
--reflection \
--input_file ./data/round1_train_data_fix.jsonl 

CUDA_VISIBLE_DEVICES=1 \
./miniconda3/envs/vllm/bin/python3 src/batch_vllm_retry.py \
--model glm-4-9b-chat  \
--model_dir models_merge \
--answer_name glm-4-9b-chat   \
--gpu_memory_utilization 0.95 \
--tp 1 \
--retry_total 5 \
--max_model_len 8192 \
--check_model \
--reflection \
--input_file ./data/round1_train_data_fix.jsonl 

CUDA_VISIBLE_DEVICES=2 \
./miniconda3/envs/vllm/bin/python3 src/batch_vllm_retry.py \
--model Qwen2-7B-Instruct  \
--model_dir models_merge \
--answer_name Qwen2-7B-Instruct   \
--gpu_memory_utilization 0.95 \
--tp 1 \
--retry_total 5 \
--max_model_len 8192 \
--check_model \
--reflection \
--input_file ./data/round1_train_data_fix.jsonl 



# model_merge_reflection
CUDA_VISIBLE_DEVICES=4 \
./miniconda3/envs/vllm/bin/python3 src/batch_vllm_retry.py \
--model Meta-Llama-3.1-8B-Instruct  \
--model_dir models_merge_reflection \
--answer_name Meta-Llama-3.1-8B-Instruct   \
--gpu_memory_utilization 0.95 \
--tp 1 \
--retry_total 5 \
--max_model_len 8192 \
--check_model \
--reflection \
--input_file ./data/round1_train_data_fix.jsonl 

CUDA_VISIBLE_DEVICES=5 \
./miniconda3/envs/vllm/bin/python3 src/batch_vllm_retry.py \
--model glm-4-9b-chat  \
--model_dir models_merge_reflection \
--answer_name glm-4-9b-chat   \
--gpu_memory_utilization 0.95 \
--tp 1 \
--retry_total 5 \
--max_model_len 8192 \
--check_model \
--reflection \
--input_file ./data/round1_train_data_fix.jsonl 

CUDA_VISIBLE_DEVICES=7 \
./miniconda3/envs/vllm/bin/python3 src/batch_vllm_retry.py \
--model Qwen2-7B-Instruct  \
--model_dir models_merge_reflection \
--answer_name Qwen2-7B-Instruct   \
--gpu_memory_utilization 0.95 \
--tp 1 \
--retry_total 5 \
--max_model_len 8192 \
--check_model \
--reflection \
--input_file ./data/round1_train_data_fix.jsonl 

CUDA_VISIBLE_DEVICES=2 \
./miniconda3/envs/vllm/bin/python3 src/batch_vllm_retry.py \
--model Qwen2.5-14B-Instruct  \
--model_dir models \
--answer_name Qwen2.5-14B-Instruct   \
--gpu_memory_utilization 0.4 \
--tp 1 \
--retry_total 5 \
--max_model_len 8192 \
--check_model \
--reflection \
--input_file ./data/round1_train_data_fix.jsonl \
--temperature 0.7 \
--top_p 0.8

CUDA_VISIBLE_DEVICES=3 \
./miniconda3/envs/vllm/bin/python3 src/batch_vllm_retry.py \
--model Qwen2.5-32B-Instruct-GPTQ-Int4  \
--model_dir models \
--answer_name Qwen2.5-32B-Instruct-GPTQ-Int4   \
--gpu_memory_utilization 0.4 \
--tp 1 \
--retry_total 5 \
--max_model_len 8192 \
--check_model \
--reflection \
--input_file ./data/round1_train_data_fix.jsonl \
--temperature 0.7 \
--top_p 0.8

############################################3












# sft+merge model check
CUDA_VISIBLE_DEVICES=0 \
./miniconda3/envs/vllm/bin/python3 src/batch_vllm_retry.py \
--model Meta-Llama-3.1-8B-Instruct,glm-4-9b-chat,Qwen2-7B-Instruct  \
--lora Meta-Llama-3.1-8B-Instruct,glm-4-9b-chat,Qwen2-7B-Instruct \
--answer_name merge_model  \
--gpu_memory_utilization 0.95 \
--tp 1 \
--retry_total 5 \
--max_model_len 8192 \
--check_model \
--input_file ./data/round1_train_data_fix.jsonl   

# reflection
CUDA_VISIBLE_DEVICES=1 \
./miniconda3/envs/vllm/bin/python3 src/batch_vllm_retry.py \
--model glm-4-9b-chat  \
--answer_name glm-4-9b-chat  \
--gpu_memory_utilization 0.95 \
--tp 1 \
--retry_total 1 \
--max_model_len 8192 \
--check_model \
--reflection \
--input_file ./data/round1_train_data_fix.jsonl  

CUDA_VISIBLE_DEVICES=2 \
./miniconda3/envs/vllm/bin/python3 src/batch_vllm_retry.py \
--model Meta-Llama-3.1-8B-Instruct  \
--answer_name Meta-Llama-3.1-8B-Instruct  \
--gpu_memory_utilization 0.95 \
--tp 1 \
--retry_total 1 \
--max_model_len 8192 \
--check_model \
--reflection \
--input_file ./data/round1_train_data_fix.jsonl  

# reflection+sft
CUDA_VISIBLE_DEVICES=1 \
./miniconda3/envs/vllm/bin/python3 src/batch_vllm_retry.py \
--model glm-4-9b-chat  \
--lora glm-4-9b-chat  \
--answer_name glm-4-9b-chat  \
--gpu_memory_utilization 0.95 \
--tp 1 \
--retry_total 1 \
--max_model_len 8192 \
--check_model \
--reflection \
--input_file ./data/round1_train_data_fix.jsonl  

CUDA_VISIBLE_DEVICES=2 \
./miniconda3/envs/vllm/bin/python3 src/batch_vllm_retry.py \
--model Meta-Llama-3.1-8B-Instruct  \
--lora Meta-Llama-3.1-8B-Instruct  \
--answer_name Meta-Llama-3.1-8B-Instruct  \
--gpu_memory_utilization 0.95 \
--tp 1 \
--retry_total 1 \
--max_model_len 8192 \
--check_model \
--reflection \
--input_file ./data/round1_train_data_fix.jsonl  

CUDA_VISIBLE_DEVICES=2 \
./miniconda3/envs/vllm/bin/python3 src/batch_vllm_retry.py \
--model Qwen2-7B-Instruct  \
--lora Qwen2-7B-Instruct  \
--answer_name Qwen2-7B-Instruct   \
--gpu_memory_utilization 0.95 \
--tp 1 \
--retry_total 1 \
--max_model_len 8192 \
--check_model \
--reflection \
--input_file ./data/round1_train_data_fix.jsonl 


# sft+reflection+merge
CUDA_VISIBLE_DEVICES=6 \
./miniconda3/envs/vllm/bin/python3 src/batch_vllm_retry.py \
--model Meta-Llama-3.1-8B-Instruct,glm-4-9b-chat,Qwen2-7B-Instruct  \
--lora Meta-Llama-3.1-8B-Instruct,glm-4-9b-chat,Qwen2-7B-Instruct \
--answer_name merge_model  \
--gpu_memory_utilization 0.95 \
--tp 1 \
--retry_total 5 \
--max_model_len 8192 \
--check_model \
--reflection \
--input_file ./data/round1_train_data_fix.jsonl 



# CUDA_VISIBLE_DEVICES=0 ./miniconda3/envs/vllm/bin/python3 src/batch_vllm_retry.py \
# --model Meta-Llama-3.1-8B-Instruct,glm-4-9b-chat,Qwen2-7B-Instruct \
# --answer_name answer  \
# --gpu_memory_utilization 1.0 \
# --tp 1 \
# --retry_total 5 \
# --max_model_len 4096 \
# --input_file /tcdata/round2_test_data.jsonl

# Infinity-Instruct-7M-Gen-Llama3_1-8B 999/1415
# Infinity-Instruct-7M-Gen-mistral-7B 951/1415
# Meta-Llama-3.1-8B-Instruct+relfection 985/1415

# base
# glm-4-9b-chat 1032/1415 5:1057
# Meta-Llama-3.1-8B-Instruct 992/1415
# qwen2 1041/1415
# Meta-Llama-3.1-8B-Instruct,glm-4-9b-chat,Qwen2-7B-Instruct 5:1172/1415=0.82

# reflection
# glm-4-9b-chat 1110/1415 5:1110/1415
# Meta-Llama-3.1-8B-Instruct 1005/1415
# Qwen2-7B-Instruct 1020/1415
# Meta-Llama-3.1-8B-Instruct,glm-4-9b-chat,Qwen2-7B-Instruct 5:1123/1415=0.79

# sft
# glm-4-9b-chat 1118/1415  5:1118
# Meta-Llama-3.1-8B-Instruct 1087/1415 
# Qwen2-7B-Instruct 1096/1415
# Meta-Llama-3.1-8B-Instruct,glm-4-9b-chat 5: 1088/1415
# Meta-Llama-3.1-8B-Instruct,glm-4-9b-chat,Qwen2-7B-Instruct 5: 1164/1415=0.82

# reflection+sft
# glm-4-9b-chat 1163/1415 5:1163
# Meta-Llama-3.1-8B-Instruct 1134/1415 
# Qwen2-7B-Instruct 1088/1415 
# Meta-Llama-3.1-8B-Instruct,glm-4-9b-chat,Qwen2-7B-Instruct-ori 5:1177/1415=0.83
# Meta-Llama-3.1-8B-Instruct,glm-4-9b-chat  5:1135/1415
# Meta-Llama-3.1-8B-Instruct,glm-4-9b-chat,Qwen2-7B-Instruct 5:1192/1415=0.84

# train reflection+sft
# glm-4-9b-chat 1134/1415
# Meta-Llama-3.1-8B-Instruct 1106/1415
# Qwen2-7B-Instruct 1091/1415 
# merge 1176/1415


# train reflection+sft tempeture=0.7
# glm-4-9b-chat 1149/1415
# Meta-Llama-3.1-8B-Instruct 1109/1415
# Qwen2-7B-Instruct 1083/1415 
# merge 


# sub best
# merge_model base
# glm-4-9b-chat 1199/1415
# Meta-Llama-3.1-8B-Instruct 1162/1415
# Qwen2-7B-Instruct 1159/1415 
# merge 1226/1415


# merge_model base temp=1.0
# glm-4-9b-chat 1188/1415
# Meta-Llama-3.1-8B-Instruct 1108/1415
# Qwen2-7B-Instruct 1128/1415 
# merge 1220/1415


# merge_model relfection 0.7
# glm-4-9b-chat 1207/1415 ; tm=0.6:1194
# Meta-Llama-3.1-8B-Instruct 1162/1415
# Qwen2-7B-Instruct 1179/1415 
# merge 1215/1415


# best
# merge_model relfection temp=1.0
# glm-4-9b-chat 1203/1415
# Meta-Llama-3.1-8B-Instruct 1166/1415
# Qwen2-7B-Instruct 1158/1415
# merge 1227/1415




# train reflection+sft+big
# Meta-Llama-3.1-8B-Instruct 1076/1415 tm=0.7:1091/1415     tm=1.0:1123/1415

# merge_reflection big
# Meta-Llama-3.1-8B-Instruct extra tm=0.3 1120/1415 tm=0.5 1104/1415 tm=0.7 1110/1415   tm=1.0 1091/1415 
# Meta-Llama-3.1-8B-Instruct tm=0.7 1107/1415; tm=0.6 1133/1415, top_p=0.9 1121/1415; tm=0.8 1112/1415; 
# glm-4-9b-chat tm=0.8 1148/1415; tm=0.7 1158/1415; tm=0.6 1168/1415; tm=-1 1152/1415
# Qwen2-7B-Instruct tm=0.7 1146/1415; tm=0.6 1105/1415; tm=0.8 1112/1415;





# default temperature & top_p
# Meta-Llama-3.1-8B-Instruct 0.6/0.9             1121/1415
# Qwen2-7B-Instruct 0.7/0.8                      1122/1415         
# glm-4-9b-chat 0.8/0.8                          1160/1415

# qwen2.5-14B 
# ref:default:1:1081/1415 
# ref:0.7-0.8:1:1066/1415  
# noref:-1:1:1136/1415 
# noref:0.7:0.8:1128/1415
# qwen系列模型不用加reflection

# lr_deepseek
# glm-4-9b-chat   
# ref:-1,-1:1:1086/1415 
# ref:0.8,0.8:1:1111/1415 
# ref:0.8,0.8:5:1180/1415
# noref:0.8,0.8:5:1173/1415
# noref:0.8,0.8:3:1158/1415
CUDA_VISIBLE_DEVICES=0 \
./miniconda3/envs/vllm/bin/python3 src/batch_vllm_retry.py \
--model glm-4-9b-chat  \
--model_dir models_merge_lr_deepseek \
--answer_name glm-4-9b-chat   \
--gpu_memory_utilization 0.95 \
--tp 1 \
--retry_total 3 \
--max_model_len 8192 \
--check_model \
--input_file ./data/round1_train_data_fix.jsonl \
--temperature 0.8 \
--top_p 0.8


# Meta-Llama-3.1-8B-Instruct 
# ref:-1,-1:1:989/1415 
# ref:0.6,0.9:1:1077/1415
# ref:0.6,0.9:5:1135/1415
# noref:0.6,0.9:5:1122/1415
CUDA_VISIBLE_DEVICES=3 \
./miniconda3/envs/vllm/bin/python3 src/batch_vllm_retry.py \
--model Meta-Llama-3.1-8B-Instruct  \
--model_dir models_merge_lr_deepseek \
--answer_name Meta-Llama-3.1-8B-Instruct   \
--gpu_memory_utilization 0.95 \
--tp 1 \
--retry_total 5 \
--max_model_len 8192 \
--check_model \
--input_file ./data/round1_train_data_fix.jsonl \
--temperature 0.6 \
--top_p 0.9




# Qwen2.5-7B-Instruct
# ref:0.7,0.8:5:1183/1415
# noref:0.7,0.8:5:1183/1415
CUDA_VISIBLE_DEVICES=2 \
./miniconda3/envs/vllm/bin/python3 src/batch_vllm_retry.py \
--model Qwen2.5-7B-Instruct  \
--model_dir models_merge_lr_deepseek \
--answer_name Qwen2.5-7B-Instruct   \
--gpu_memory_utilization 0.95 \
--tp 1 \
--retry_total 5 \
--max_model_len 8192 \
--check_model \
--input_file ./data/round1_train_data_fix.jsonl \
--reflection \
--temperature 0.7 \
--top_p 0.8










# Qwen2.5-Math-7B-Instruct
# ref:0.7,0.8:5:986/1415
# noref:0.7,0.8:5:637/1415
# train:ref:0.7,0.8:5:890
# train:noref:0.7,0.8:5:1091
CUDA_VISIBLE_DEVICES=1 \
./miniconda3/envs/vllm/bin/python3 src/batch_vllm_retry.py \
--model Qwen2.5-Math-7B-Instruct  \
--model_dir models_merge_lr_deepseek \
--answer_name Qwen2.5-Math-7B-Instruct   \
--gpu_memory_utilization 0.95 \
--tp 1 \
--retry_total 5 \
--max_model_len 4096 \
--check_model \
--input_file ./data/round1_train_data_fix.jsonl \
--reflection \
--temperature 0.7 \
--top_p 0.8


# Qwen2.5-Coder-7B
# ref:0.7,0.8:5:724/1415
# noref:0.7,0.8:5:697/1415
# train:ref:0.7,0.8:5:1025/1415
# train:noref:0.7,0.8:5:805/1415
CUDA_VISIBLE_DEVICES=3 \
./miniconda3/envs/vllm/bin/python3 src/batch_vllm_retry.py \
--model Qwen2.5-Coder-7B  \
--model_dir models_merge_lr_deepseek \
--answer_name Qwen2.5-Coder-7B   \
--gpu_memory_utilization 0.95 \
--tp 1 \
--retry_total 5 \
--max_model_len 4096 \
--check_model \
--input_file ./data/round1_train_data_fix.jsonl \
--reflection \
--temperature 0.7 \
--top_p 0.8


# Qwen2.5-32B-Instruct-GPTQ-Int4
# ref:0.7,0.8:5:1091/1415
# noref:0.7,0.8:5:1084/1415
CUDA_VISIBLE_DEVICES=4 \
./miniconda3/envs/vllm/bin/python3 src/batch_vllm_retry.py \
--model Qwen2.5-32B-Instruct-GPTQ-Int4  \
--model_dir models \
--answer_name Qwen2.5-32B-Instruct-GPTQ-Int4   \
--gpu_memory_utilization 0.4 \
--tp 1 \
--retry_total 5 \
--max_model_len 4096 \
--check_model \
--input_file ./data/round1_train_data_fix.jsonl \
--reflection \
--temperature 0.7 \
--top_p 0.8







# Meta-Llama-3.1-70B-Instruct 
# ref:0.6,0.9:5:1240/1415                
# noref:0.6,0.9:5:1248/1415              1248/1415
CUDA_VISIBLE_DEVICES=6,7 \
./miniconda3/envs/vllm/bin/python3 src/batch_vllm_retry.py \
--model Meta-Llama-3.1-70B-Instruct  \
--model_dir models_merge_lr_deepseek \
--answer_name Meta-Llama-3.1-70B-Instruct   \
--gpu_memory_utilization 0.95 \
--tp 2 \
--retry_total 5 \
--max_model_len 8192 \
--check_model \
--input_file ./data/round1_train_data_fix.jsonl \
--reflection \
--temperature 0.6 \
--top_p 0.9
# Meta-Llama-3.1-70B-Instruct-INT2 
# noref:0.6,0.9:5:1248/1415              1248/1415
CUDA_VISIBLE_DEVICES=5 \
./miniconda3/envs/vllm/bin/python3 src/batch_vllm_retry.py \
--model Meta-Llama-3.1-70B-Instruct  \
--model_dir models_merge_lr_deepseek_quant \
--answer_name Meta-Llama-3.1-70B-Instruct   \
--gpu_memory_utilization 0.95 \
--tp 1 \
--retry_total 5 \
--max_model_len 1024 \
--check_model \
--input_file ./data/round1_train_data_fix.jsonl \
--temperature 0.6 \
--top_p 0.9 \
--cpu_offload_gb 10

########################
# Qwen2.5-14B-Instruct
# ref:0.7,0.8:5:1231/1415
# noref:0.7,0.8:5:1236/1415     
CUDA_VISIBLE_DEVICES=3 \
./miniconda3/envs/vllm/bin/python3 src/batch_vllm_retry.py \
--model Qwen2.5-14B-Instruct  \
--model_dir models_merge_lr_deepseek \
--answer_name Qwen2.5-14B-Instruct   \
--gpu_memory_utilization 0.95 \
--tp 1 \
--retry_total 5 \
--max_model_len 8192 \
--check_model \
--input_file ./data/round1_train_data_fix.jsonl \
--reflection \
--temperature 0.7 \
--top_p 0.8
########################
# Qwen2.5-32B-Instruct 
# deepseek
# ref:0.7,0.8:5:1238    1e-4:1265/1415
# noref:0.7,0.8:5:1258  1e-4:1246/1415
CUDA_VISIBLE_DEVICES=7 \
./miniconda3/envs/vllm/bin/python3 src/batch_vllm_retry.py \
--model Qwen2.5-32B-Instruct  \
--model_dir models_merge_lr_deepseek \
--answer_name Qwen2.5-32B-Instruct   \
--gpu_memory_utilization 0.95 \
--tp 1 \
--retry_total 5 \
--max_model_len 4096 \
--check_model \
--input_file ./data/round1_train_data_fix.jsonl \
--temperature 0.7 \
--top_p 0.8
########################
# Qwen2.5-32B-Instruct int4
# deepseek
# ref:0.7,0.8:5:1249/1415           1e-4:1255
# noref:0.7,0.8:5:1257/1415=0.88    1e-4:1262
CUDA_VISIBLE_DEVICES=0 \
./miniconda3/envs/vllm/bin/python3 src/batch_vllm_retry.py \
--model Qwen2.5-32B-Instruct  \
--model_dir models_merge_lr_deepseek_quant \
--answer_name Qwen2.5-32B-Instruct   \
--gpu_memory_utilization 1 \
--tp 1 \
--retry_total 5 \
--max_model_len 4096 \
--check_model \
--input_file ./data/round1_train_data_fix.jsonl \
--temperature 0.7 \
--top_p 0.8
######################################
# llama3.1-405b
# ref:0.7,0.8:5:   1258/1415
# noref:0.7,0.8:5: 1265/1415      
CUDA_VISIBLE_DEVICES=5 \
./miniconda3/envs/vllm/bin/python3 src/batch_vllm_retry.py \
--model Qwen2.5-32B-Instruct  \
--model_dir models_merge_lr_llama31_405b_quant \
--answer_name Qwen2.5-32B-Instruct   \
--gpu_memory_utilization 1 \
--tp 1 \
--retry_total 5 \
--max_model_len 4096 \
--check_model \
--input_file ./data/round1_train_data_fix.jsonl \
--reflection \
--temperature 0.7 \
--top_p 0.8

########################
# Qwen2.5-72B-Instruct
# ref:0.7,0.8:5:1251/1415=0.884
# noref:0.7,0.8:5:1262/1415=0.89
CUDA_VISIBLE_DEVICES=4,5 \
./miniconda3/envs/vllm/bin/python3 src/batch_vllm_retry.py \
--model Qwen2.5-72B-Instruct  \
--model_dir models_merge_lr_deepseek \
--answer_name Qwen2.5-72B-Instruct   \
--gpu_memory_utilization 0.95 \
--tp 2 \
--retry_total 5 \
--max_model_len 8192 \
--check_model \
--input_file ./data/round1_train_data_fix.jsonl \
--temperature 0.7 \
--top_p 0.8 
# Qwen2.5-72B-Instruct-INT2
# ref:0.7,0.8:5:
# noref:0.7,0.8:5:
CUDA_VISIBLE_DEVICES=4 \
./miniconda3/envs/vllm/bin/python3 src/batch_vllm_retry.py \
--model Qwen2.5-72B-Instruct  \
--model_dir models_merge_lr_deepseek_quant \
--answer_name Qwen2.5-72B-Instruct   \
--gpu_memory_utilization 0.95 \
--tp 1 \
--retry_total 1 \
--max_model_len 1024 \
--check_model \
--input_file ./data/round1_train_data_fix.jsonl \
--temperature 0.7 \
--top_p 0.8 \
--cpu_offload_gb 10



# merge
CUDA_VISIBLE_DEVICES=0 \
./miniconda3/envs/vllm/bin/python3 src/batch_vllm_retry.py \
--model Qwen2.5-32B-Instruct-deepseek,Qwen2.5-32B-Instruct-llama31_405b  \
--model_dir models_quant \
--answer_name Qwen2.5-32B-Instruct \
--gpu_memory_utilization 1 \
--tp 1 \
--retry_total 1 \
--max_model_len 4096 \
--check_model \
--input_file ./data/round1_train_data_fix.jsonl \
--temperature 0.7 \
--top_p 0.8 \
--topk 1


CUDA_VISIBLE_DEVICES=0 ./miniconda3/envs/vllm/bin/python3 src/batch_vllm_retry.py \
--model Qwen2.5-32B-Instruct-deepseek,Qwen2.5-32B-Instruct-llama31_405b  \
--model_dir models_quant \
--answer_name answer  \
--gpu_memory_utilization 1.0 \
--tp 1 \
--retry_total 3 \
--max_model_len 4096 \
--input_file /tcdata/round2_test_data.jsonl \
--temperature 0.7 \
--top_p 0.8



CUDA_VISIBLE_DEVICES=0 ./miniconda3/envs/vllm/bin/python3 src/batch_vllm_retry.py \
--model Qwen2.5-32B-Instruct-llama31_405b  \
--model_dir models_quant \
--answer_name answer  \
--gpu_memory_utilization 0.4 \
--tp 1 \
--retry_total 5 \
--max_model_len 4096 \
--input_file ./data/round1_test_data.jsonl \
--temperature 0.7 \
--top_p 0.8 \
--topk 1


CUDA_VISIBLE_DEVICES=0 ./miniconda3/envs/vllm/bin/python3 src/batch_vllm_retry.py \
--model Qwen2.5-32B-Instruct-deepseek,Qwen2.5-32B-Instruct-llama31_405b  \
--model_dir models_quant \
--answer_name answer  \
--gpu_memory_utilization 0.4 \
--tp 1 \
--retry_total 3 \
--max_model_len 4096 \
--input_file ./data/round1_test_data.jsonl \
--temperature 0.7 \
--top_p 0.8 \
--topk 1