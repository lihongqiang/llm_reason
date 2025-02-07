# 方案说明文档
我们基于qwen2.5-32b做为base模型训练lora模型，在4bit量化模型压缩下，满足32G的显存大小限制。
主要流程包括以下几个方面：
- 数据处理: 我们对初赛的数据进行人为校验和修正。
- 训练数据：对修正后的数据，用DeepSeek-Coder-V2-Instruct和Meta-Llama-3.1-405B-Instruct-FP8做为teacher模生成sft训练数据。
- prompt：我们采用thinking+reflection的方式，让模型先思考推理过程，再反思结果，如果不对再修正，最终返回选项。
- 模型训练：我们用LLamaFactory做模型训练，合并和量化。
- 模型评估：将qwen2.5-32b+llama3.1-405b数据和qwen2.5-32b+deepseek数据训练后的量化模型做ensemble，得到最终结果。

## 数据处理
在通过开源模型跑训练数据，对无法预估答案的问题进行整理，然后人为判断是否是bad case，答案是否错误并进行修正。  
通过修正round1_train_data.jsonl，我们得到新的训练数据文件data/round1_train_data_fix.jsonl。

## 训练数据
因为训练数据的答案是选项，并不是完整的回答。因此这里采用更大的模型辅助输出推理过程完成回答。我们结合以下两种方式进行实现: 
- 数据重复尝试生成：重复调用teacher模型，直到得到正确答案为止，用正确的推理过程构建训练数据。对剩下的teacher模型多次尝试也不能推理出结果的数据再采用以下，已知答案生成过程的方式处理。
- 已知答案生成过程：采用teacher模型，设置prompt，已知答案，写出分析过程，最终得到答案。 
这里以DeepSeek-Coder-V2-Instruct为例子，代码在run_batch_gen_data.sh中。  

最终得到的数据存放到src/LLaMA-Factory/data/路径下。
```bash
# 数据重复尝试生成
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 src/batch_vllm_gen_data.py \
--model DeepSeek-Coder-V2-Instruct \
--input_file ./data/round1_train_data_fix.jsonl \
--gpu_memory_utilization 0.98 \
--retry_total 20 \
--max_model_len 8192 \
--tp 8 \
--reflection \
--temperature 0.7 \
--top_p 0.95 

# 已知答案生成过程
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 src/batch_vllm_gen_data.py \
--model DeepSeek-Coder-V2-Instruct \
--input_file ./data/round1_train_data_fix.jsonl \
--gpu_memory_utilization 0.98 \
--retry_total 20 \
--max_model_len 8192 \
--tp 8 \
--reflection \
--temperature 0.7 \
--top_p 0.95 \
--build_answer 
```

## prompt
我们采用thinking+reflection的prompt如下：
```python
def get_prompt_reflection(problem, question, options):
    options = '\n'.join(f"{'ABCDEFG'[i]}. {o}" for i, o in enumerate(options))
    prompt = f"""你是一个逻辑推理专家，擅长复杂的推理和反思。以下是一个单项选择题，所有的问题都是（close-world assumption）闭世界假设，即未观测事实都为假。请逐步分析问题，在<thinking>标签内描述推理过程，在<reflection>标签内检查推理过程是否正确，如果有误及时纠错，在<output>标签内提供你的最终回答。输出格式为:<output>因此答案是：A</output>。题目如下：

        ### 题目:
        {problem}

        ### 问题:
        {question}
        {options}
        """
    return prompt
```

## 模型训练
本项目采用llamafactory框架进行lora训练，首先clone LLaMA-Factory代码库，本代码库中只包含了改动的代码。
```
git clone https://github.com/hiyouga/LLaMA-Factory.git
```

### 增加数据文件
修改LLaMa-Factory/data/dataset_info.json
```json
# 增加数据集
"Meta-Llama-3.1-405B-Instruct-FP8_sft_fix":{
    "file_name": "Meta-Llama-3.1-405B-Instruct-FP8_sft_fix.jsonl"
  },
  "DeepSeek-Coder-V2-Instruct_sft_fix":{
    "file_name": "DeepSeek-Coder-V2-Instruct_sft_fix.jsonl"
  },
```
### 增加量化校验数据集
data/round1_text.json

### 模型训练，合并和压缩
增加模型训练，合并和量化的配置
- examples/train_lora/qwen2.5_32b_quant.yaml
- examples/merge_lora/qwen2.5_32b_sft.yaml
- examples/merge_lora/qwen2.5_32b_quant.yaml
调用llamafactory开始训练，这里使用8卡A100训练。
```
# sft训练
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
llamafactory-cli train examples/train_lora/qwen2.5-32B.yaml

# merge lora
llamafactory-cli export examples/merge_lora/qwen2.5_32b_sft.yaml

# quant
CUDA_VISIBLE_DEVICES=0,1,2,3 \
llamafactory-cli export examples/merge_lora/qwen2.5_32b_quant.yaml
```

### 模型评估
新建models_quant，把合并两个模型放到文件夹内，做模型ensemble。
```bash
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
--top_p 0.8
```

## 代码与文件结构
```
├── README.md
├── data
│   └── round1_train_data_fix.jsonl  # 原始数据
├── run_batch_gen_data.sh # 训练数据生成脚本
├── run_batch_retry.sh # 模型ensemble推理脚本
├── run_train_merge_quant.sh # 模型训练，合并，量化脚本
└── src
    ├── LLaMA-Factory
    │   ├── data
    │   │   ├── DeepSeek-Coder-V2-Instruct_sft_fix.jsonl # DeepSeek-Coder-V2-Instruct的teacher模型生成的训练数据
    │   │   ├── Meta-Llama-3.1-405B-Instruct-FP8_sft_fix.jsonl # Meta-Llama-3.1-405B-Instruct的teacher模型生成的训练数据
    │   │   ├── dataset_info.json # 数据集配置
    │   │   └── round1_text.json # 量化校验数据
    │   └── examples
    │       ├── merge_lora
    │       │   ├── qwen2.5_32b_quant.yaml # 模型量化配置
    │       │   └── qwen2.5_32b_sft.yaml # 模型合并配置
    │       └── train_lora
    │           └── qwen2.5-32B.yaml # 模型训练配置
    ├── batch_vllm_gen_data.py # 训练数据生成脚本
    └── batch_vllm_retry.py # 模型推理脚本
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