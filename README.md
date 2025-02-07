# 方案说明文档
我们基于llama3.1-70b训练lora模型，在非量化压缩情况下，采用bf16类型加载，显存140G，达到**0.8788**的效果，打平测试的所有开源最好模型llama3.1-405b的效果，因初赛显存限制已删除记录。  
我们训练的llama3.1-70b+lora模型在3bit量化模型压缩下，满足32G的显存大小限制，并达到**0.8524**的效果，在4bit量化压缩下，满足40G的显存大小限制，达到**0.8539**的效果。  
以下是所有模型在测试集上的结果。

模型|参数量|显存|训练数据|测试数据
----|---|---|---|---
llama3.1|405b|200G|0.8810|0.8788
llama3.1+lora|70b|132G|-|**0.8788**
llama3.1+lora+3bit|70b|30G|-|**0.8524**
llama3.1+lora+4bit|70b|38G|-|**0.8539**

## 方法介绍
我们以**llama3.1+lora+3bit**开始介绍，4bit的版本只需要把模型压缩部分的配置文件从3bit改成4bit即可，llama3.1+lora的版本只需要不做量化处理，直接对合并后的模型进行推理即可。  
我们的整体方法流程包含以下几个部分：
- 数据处理
- base模型选择
- base模型SFT训练
- base+lora模型合并
- 模型压缩
- 模型评估

### 数据处理
#### 训练数据异常过滤与处理
在通过开源模型跑训练数据，对无法预估答案的问题进行整理，然后人为判断是否是bad case，答案是否错误并进行修正。  
通过修正round1_train_data.jsonl，我们得到新的训练数据文件round1_train_data_fix.jsonl。

#### 模型训练数据构建
因为训练数据的答案是选项，并不是完整的回答。因此这里采用模型辅助输出完成回答。我们采用了两种方式进行实现: 

- 蒸馏模型：采用更大的模型，设置prompt，已知答案，写出分析过程，最终得到答案。 
- 自反思模型：只考虑首次错误的题，重复调用base模型，直到得到正确答案为止，用正确的推理过程构建训练数据。  
  
我们对比了采用蒸馏模型全部SFT数据和做错题的数据的自反思模型，结果发现错题对于模型来说效果更佳明显。最终我们选择自反思模型，只对首次错误的数据，多次调用base模型得到正确答案的推理过程，构建训练数据。  
当然这里也可以基于base模型采用两轮迭代，对于答错的题目，再次调用模型根据当前推理过程和正确答案，进行分析得到最终正确的推理过程。这里没有继续尝试。

#### 部署base模型vllm服务：  
```CUDA_VISIBLE_DEVICES=0 \   
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
--port 8000   
```
#### 调用api接口生成初始答案
```
# generate upload.jsonl upload_merge.jsonl gen_dpo.jsonl error.jsonl
python3 ./src/base_llm_api_args.py \
--model Meta-Llama-3.1-70B-Instruct \
--stage gen_dpo \
--mode gen_dpo \
--input_file ./data/round1_train_data_fix.jsonl \
--eval \
--port 8000
```

#### 合并题目的多个问题
```
# generate upload_merge_base.json answer.jsonl
python3 test_merge_base.py \
--model Meta-Llama-3.1-70B-Instruct \
--stage gen_dpo \
--input_file ./data/upload_merge.jsonl \
--base_file ./data/round1_train_data_fix.jsonl 
```

#### 生成最终SFT训练数据
调用api接口对错误的问题进行多次请求，直到得到问题得到正确答案。
```
# generate gen_dpo_true.jsonl
python3 ./src/base_llm_api_args.py \
--model Meta-Llama-3.1-70B-Instruct \
--stage gen_dpo \
--mode gen_dpo \
--eval \
--input_file ./data/upload_merge_base.jsonl \
--port 8000 \
--try_true
```
最后对gen_dpo_true.jsonl首尾增加中括号[]，得到最终模型训练数据round1_train_llama3.1_sft_self.jsonl。
```
train_data_path=./data/round1_train_llama3.1_sft_self_test.jsonl
rm $train_data_path
touch $train_data_path
echo '[' > $train_data_path
cat ./data/gen_dpo_true.jsonl >> $train_data_path
echo ']' >> $train_data_path
```

### base模型选择
我们评估了常用的开源模型的效果。基于相同的prompt，我们发现一般而言，模型越大，效果越明显。最终在结合初赛要求40G显存和复赛32G显存的基础上，最终选择llama3.1-70b作为我们的base模型。
模型|大小|训练数据|测试数据
----|---|---|---
qwen2|7b|0.7469|0.7342
llama3.1|8b|0.7353|0.7583
deepseek-code-v2|16b|0.7621|0.7711
gemma2|27b|0.81|0.8238
yi|34b|-|0.7041
llama3.1|70b|0.8620|**0.8705**
qwen2|72b|0.8613|0.8667
qwen2-math|72b|-|0.8389
llama3.1|405b|0.8810|**0.8788**

### base模型SFT训练
我们分别尝试了SFT和DPO的效果，最终选择对llama3.1-70b模型，使用训练数据round1_train_llama3.1_sft_self.jsonl，进行SFT+QLora训练得到Lora模型。
本项目采用llamafactory框架进行Qlora训练，首先clone LLaMA-Factory代码库，本代码库中只包含了改动的代码。
```
git clone https://github.com/hiyouga/LLaMA-Factory.git
```

#### 增加数据文件
修改LLaMa-Factory/data/dataset_info.json
``` 
# copy the file into the data dir and add config in data_info.json
sed -i '1a "round1_train_llama3.1_sft_self\":{\"file_name\": \"./data/round1_train_llama3.1_sft_self_test.jsonl\"},' ./src/LLaMA-Factory/data/dataset_info.json

```
#### 设置训练配置
增加LLaMA-Factory/examples/train_lora/llama3.1-70b_self.yaml
```
### model
model_name_or_path: LLM-Research/Meta-Llama-3.1-70B-Instruct
quantization_bit: 4
quantization_method: bitsandbytes

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all

### dataset
dataset: round1_train_llama3.1_sft_self
template: llama3
cutoff_len: 10240
max_samples: 10000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: ./models/llama3.1-70b/lora/sft
logging_steps: 5
save_steps: 20
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 1
learning_rate: 5.0e-5
num_train_epochs: 10.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
lora_rank: 16
lora_alpha: 32
flash_attn: auto

### eval
val_size: 0.01
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 10
```
#### 开始模型训练
调用llamafactory开始训练，这里使用8卡A100训练。
```
CUDA_LAUNCH_BLOCKING=1 \
TF_ENABLE_ONEDNN_OPTS=0 \
OMP_NUM_THREADS=16 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
llamafactory-cli train examples/train_lora/llama3.1-70b_self.yaml
```

### base+lora模型合并
#### 设置模型合并配置
对llama3.1-70b+Lora模型进行合并，得到llama3.1-70b-lora模型。  
增加LLaMA-Factory/examples/merge_lora/llama3.1_lora_sft.yaml
```
### model
model_name_or_path: LLM-Research/Meta-Llama-3.1-70B-Instruct
adapter_name_or_path: models/llama3.1-70b/lora/sft
template: llama3
finetuning_type: lora

### export
export_dir: models/llama3.1_70b_lora_sft
export_size: 4
export_device: cpu
export_legacy_format: true
```
#### 开始模型合并
调用llamafactory开始训练，这里使用8卡A100训练。对合并后的模型直接进行推理，即可得到0.8788的结果。
```
llamafactory-cli export examples/merge_lora/llama3.1_lora_sft.yaml
```

### 模型压缩
#### 构建模型压缩校验数据集
根据训练和测试数据round1_test_data.jsonl和round1_train_data.jsonl生成prompt数据集，用于校验量化模型。
```
# generate data/round1_text.json
python3 build_quant_val_data.py
```

#### 设置模型压缩脚本
对llama3.1-70b-lora模型采用Auto-gptq进行3bit压缩  
增加examples/merge_lora/llama3.1_quant_bf16.yaml文件
```
### model
model_name_or_path: models/llama3.1_70b_lora_sft
template: llama3

### export
export_dir: models/llama3.1_70b_lora_3bit_bf16_text
export_quantization_bit: 3
export_quantization_dataset: data/round1_text.json
export_quantization_maxlen: 256
export_quantization_nsamples: 1024
export_size: 4
export_device: cpu
export_legacy_format: true
infer_dtype: bfloat16
use_cache: false
print_param_status: true
```
#### 开始模型压缩
```
CUDA_VISIBLE_DEVICES=0,1,2 \
TF_ENABLE_ONEDNN_OPTS=0 \
llamafactory-cli export examples/merge_lora/llama3.1_quant_bf16.yaml
```

### 模型评估
#### 部署sft模型vllm api服务
这里采用单卡A100运行，因为复赛显存大小限制为32G，这里限制0.4的显存使用率，即0.4*80=32G。
```CUDA_VISIBLE_DEVICES=0 \
python -m vllm.entrypoints.openai.api_server  \
--model models/llama3.1_70b_lora_3bit_bf16_text  \
--served-model-name llama3.1_70b_lora_3bit_bf16_text \
--max-model-len=4096 \
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
```

#### 调用api生成SFT模型对应的答案
```
# generate upload.jsonl upload_merge.jsonl
python3 base_llm_api_args.py \
--model llama3.1_70b_lora_3bit_bf16_text \
--stage test \
--mode infer \
--input_file ./data/round1_test_data.jsonl \
--port 8000
```

#### 合并base文件生成answer文件
```
# generate upload_merge_base.json answer.jsonl
python3 test_merge_base.py \
--model Meta-Llama-3.1-70B-Instruct \
--stage test \
--input_file ./data/upload_merge.jsonl \
--base_file ./data/round1_test_data.jsonl 
```

## 代码与文件结构
```
├── README.md
├── data
│   ├── error.jsonl # 中间结果：记录训练数据集推理的正确率
│   ├── external_data # 额外数据文件夹：无
│   ├── gen_dpo.jsonl # 中间结果：记录base模型评估训练数据生成的prompt,output和answer
│   ├── gen_dpo_true.jsonl # 中间结果：记录对首次回答错误的数据，反复调用模型自反思得到的结果，用于训练SFT模型
│   ├── round1_test_data.jsonl # 测试数据
│   ├── round1_train_data.jsonl # 训练数据
│   ├── round1_train_data_fix.jsonl # 中间结果：在跑开源模型过程中，查看bad case找到无法回答和答案错误的问题，进行修复后的结果
│   ├── round1_train_llama3.1_sft_self.jsonl # gen_dpo_true.jsonl复制后的文件，用于训练SFT
│   ├── upload.jsonl # 记录模型生成的每个问题的答案
│   ├── upload_merge.jsonl # 合并每个题目所有问题的结果，只记录了有回答的结果
│   └── upload_merge_base.jsonl # upload_merge.jsonl与提问的原始文件对比，合并所有的问题，对没有回答的问题不设置answer字段
├── requirements.txt # 环境依赖
├── run.sh # 入口程序，对训练好的模型部署api服务，调用api接口获取结果文件
├── src
│   ├── LLaMA-Factory # llamafactory代码库中修改的代码部分
│   │   ├── data
│   │   │   ├── dataset_info.json # 数据集配置文件
│   │   │   ├── round1_text.json # 量化时校验的文件
│   │   │   └── round1_train_llama3.1_sft_self.jsonl # 训练数据文件
│   │   ├── examples
│   │   │   ├── merge_lora
│   │   │   │   ├── llama3.1_lora_sft.yaml # 模型base+lora合并的配置
│   │   │   │   └── llama3.1_quant_bf16.yaml # 模型3bit量化配置
│   │   │   └── train_lora
│   │   │       └── llama3.1-70b_self.yaml # 型SFT训练配置
│   │   └── models # 保存生成的模型文件夹
│   ├── base_llm_api_args.py # 调用vllm api服务的代码文件脚本
│   ├── build_quant_val_data.py # 构建round1_text.json量化校验文件的脚本
│   └── test_merge_base.py # 合并原始文件和模型评估的结果，对未评估的问题不设置answer字段，生成最终answer文件的脚本
└── submit
    ├── answer_20240814_040506.jsonl # llama3.1+lora+merge后的结果
    ├── answer_20240815_040506.jsonl # llama3.1+lora+3bit结果
    └── answer_20240815_050507.jsonl # llama3.1+lora+4bit结果
```
## 环境依赖
python环境依赖包在requirements.txt文件中，下面列出主要的环境参数。
环境|参数
----|---
内核版本|Linux 6b9d144d4892 5.4.0-144-generic #161-Ubuntu SMP Fri Feb 3 14:49:04 UTC 2023 x86_64 x86_64 x86_64 GNU/Linux
操作系统版本|Ubuntu 22.04.3 LTS
GPU Driver Version|535.183.01
CUDA Version|12.2
cuDNN | v8.9.7
GPU|8*A100
python版本|Python 3.10.14
flash-attn|2.6.3
torch|2.3.1
transformers|4.43.1
llamafactory|0.8.4.dev0