from vllm import LLM, SamplingParams
from tqdm import tqdm
import argparse
import json
import re
import torch

# 这里使用extract抽取模获得抽取的结果，如果不在可选范围内，选最后一个
def extract(input_text, options=['A', 'B', 'C', 'D', 'E', 'F', 'G']):
    try:
        problems = re.findall(r"答案是[:： ,]*([A-G])", input_text.strip('\n'))
        if not problems or len(problems)<=0:
            # print(f'[ERROR] input_text:{input_text} answer: {problems}')
            return 'NULL'
        for problem in problems[::-1]:
            if problem in options:
                return problem
        # print(f'[ERROR] input_text:{input_text} answer: {problems}')
        return 'NULL'
    except Exception as e:
        print(f'extract error: {e}')
        return 'NULL'
    
def get_prompt(problem, question, options):
    options = '\n'.join(f"{'ABCDEFG'[i]}. {o}" for i, o in enumerate(options))
    prompt = f"""你是一个逻辑推理专家，擅长解决逻辑推理问题。以下是一个逻辑推理的题目，形式为单项选择题。所有的问题都是（close-world assumption）闭世界假设，即未观测事实都为假。请逐步分析问题选择最正确的答案并在最后一行输出，最后一行的格式为"因此答案是：A"。题目如下：

        ### 题目:
        {problem}

        ### 问题:
        {question}
        {options}
        """

    # ret = f"""
    #     <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    #     <|start_header_id|>user<|end_header_id|>

    #     {prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    # """
    return prompt

def get_prompt_system(problem, question, options):
    options = '\n'.join(f"{'ABCDEFG'[i]}. {o}" for i, o in enumerate(options))
    system_prompt = "You are a world-class AI system, capable of complex reasoning and reflection. Reason through the query inside <thinking> tags, and then provide your final response inside <output> tags. If you detect that you made a mistake in your reasoning at any point, correct yourself inside <reflection> tags."
    prompt = f"""
        你是一个逻辑推理专家，擅长解决逻辑推理问题。以下是一个逻辑推理的题目，形式为单项选择题。所有的问题都是（close-world assumption）闭世界假设，即未观测事实都为假。请逐步分析问题选择最正确的答案并在最后一行输出，最后一行的格式为"因此答案是：A"。题目如下：

        ### 题目:
        {problem}

        ### 问题:
        {question}
        {options}
        """
    ret_prompt = f"""
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    return ret_prompt

# 合并问题
def merge_problems(data, answer_name):
    problems = {}
    for item in data:
        problem = item['id']
        if problem in problems:
            exist_problem = problems[problem]
            for id, question in enumerate(item['questions']):
                if answer_name not in exist_problem['questions'][id] and answer_name in item['questions'][id]:
                    exist_problem['questions'][id][answer_name] = item['questions'][id][answer_name]
        else:
            problems[problem] = item
            
    return problems.values()

def write_file(file, data):
    with open(file, 'w', encoding='utf-8') as writer:
        for sample in data:
            writer.write(json.dumps(sample, ensure_ascii=False))
            writer.write('\n')

def get_outputs(llm, prompts, sampling_params, data, prompt_item_idx, answer_name):
    # 计算结果
    outputs = llm.generate(prompts, sampling_params=sampling_params, )

    # 统计
    result_cnt = 0
    left_prompts = []
    left_prompt_item_idx = []
    for idx, output in enumerate(outputs):
        prompt = output.prompt
        res = output.outputs[0].text
        result = extract(res)
        pid, id = prompt_item_idx[idx]
        question = data[pid]['questions'][id]
        if result != 'NULL':
            result_cnt += 1
            question[answer_name] = result
        else:
            left_prompts.append(prompt)
            left_prompt_item_idx.append((pid, id))

        # print(f"Prompt: {prompt!r},\n Generated text: {res!r}\n")
        # print(f"question: {question}")
    print(f'result: {result_cnt}/{len(outputs)}')
    return result_cnt!=len(outputs), left_prompts, left_prompt_item_idx

def check_answer(data, model):
    true_cnt = 0
    total_cnt = 0
    for item in data:
        for question in item['questions']:
            if model in question and question['answer'] == question[model]:
                true_cnt += 1
            total_cnt += 1
    print(f"true_cnt/total={true_cnt}/{total_cnt}")

def fill_answer(data, prompt_item_idx, answer_name):
    for pid, id in prompt_item_idx:
        question = data[pid]['questions'][id]
        question[answer_name] = 'A'
        print(f'fill answer: {pid}, {id}, A')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='LLM API exec')
    parser.add_argument('--model', type=str, help='model name')
    parser.add_argument('--answer_name', type=str, help='answer name')
    parser.add_argument('--input_file', type=str, help='输入文件')
    parser.add_argument('--topk', type=int, default=0, help='测试程序是否正常')
    parser.add_argument('--gpu_memory_utilization', type=float, default=1.0, help='gpu使用率')
    parser.add_argument('--tp', type=int, default=1, help='多少卡')
    parser.add_argument('--max_model_len', type=int, default=8192, help='max_model_len')
    parser.add_argument('--check_model', action='store_true', help='检查模型效果')
    
    args = parser.parse_args()
    print(f'start args: {args}')

    # 设置
    model = args.model # 'deepseek-code-v2'
    input_file = args.input_file # '/data/root/jupyter/modelscope/round1_train_data.jsonl'
    topk = args.topk
    answer_name = args.answer_name

    # 输出文件
    output_dir = f'./data'
    result_file = f'./results.jsonl'

    # 按行读取数据
    data = []
    with open(input_file) as reader:
        for line in reader:
            sample = json.loads(line)
            data.append(sample)
    if topk>0:
        data = data[:topk]

    

    # GLM-4-9B-Chat-1M
    # max_model_len, tp_size = 1048576, 4

    # GLM-4-9B-Chat
    # 如果遇见 OOM 现象，建议减少max_model_len，或者增加tp_size
    max_model_len, tp_size = args.max_model_len, args.tp
    # model_name = "glm-4-9b-chat"
    # prompt = [{"role": "user", "content": "你好"}]

    # tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    llm, sampling_params = None, None
    if model in ['GLM-4-9B-Chat', 'glm-4-9b-chat-sft']:
        llm = LLM(
            model=f"./models/{model}",
            tokenizer=f'./models/{model}', 
            trust_remote_code=True,
            tensor_parallel_size=tp_size,
            max_model_len=max_model_len,
            enforce_eager=True,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype=torch.float16,
            # GLM-4-9B-Chat-1M 如果遇见 OOM 现象，建议开启下述参数
            # enable_chunked_prefill=True,
            # max_num_batched_tokens=8192
        )
        stop_token_ids = [151329, 151336, 151338]
        sampling_params = SamplingParams(temperature=0.7, max_tokens=max_model_len, stop_token_ids=stop_token_ids)
    elif model in ['llama3.1_70b_lora_3bit_bf16_text', 'DeepSeek-Coder-V2-Lite-Instruct']:
        sampling_params = SamplingParams(n=1, best_of=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, 
                                     temperature=0.7, top_p=1.0, top_k=-1, min_p=0.0, seed=None, use_beam_search=False, length_penalty=1.0, 
                                     early_stopping=False, stop=[], stop_token_ids=[], include_stop_str_in_output=False, ignore_eos=False, 
                                     max_tokens=None, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, 
                                     spaces_between_special_tokens=True, truncate_prompt_tokens=None)
    
        llm = LLM(model=f"./models/{model}",
                tokenizer=f'./models/{model}', 
                gpu_memory_utilization=args.gpu_memory_utilization,
                skip_tokenizer_init=False, tokenizer_mode='auto', 
                revision=None, rope_scaling=None, rope_theta=None, 
                tokenizer_revision=None, 
                trust_remote_code=True, 
                dtype=torch.float16, 
                max_model_len=max_model_len,
                download_dir=None, 
                tensor_parallel_size=tp_size, 
                pipeline_parallel_size=tp_size, 
                disable_custom_all_reduce=False, 
                # quantization='auto', 
                enforce_eager=True, 
                kv_cache_dtype='auto',
                quantization_param_path=None,
                use_v2_block_manager=False, enable_prefix_caching=False)
    else:
        llm = LLM(
            model=f"./models/{model}",
            tokenizer=f'./models/{model}', 
            trust_remote_code=True,
            tensor_parallel_size=tp_size,
            max_model_len=max_model_len,
            enforce_eager=True,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype=torch.float16,
        )
        sampling_params = SamplingParams(temperature=0.7, max_tokens=max_model_len, top_p=0.95)

    # 生成prompt
    prompts = []
    prompt_item_idx = []
    # system_prompt = "You are a world-class AI system, capable of complex reasoning and reflection. Reason through the query inside <thinking> tags, and then provide your final response inside <output> tags. If you detect that you made a mistake in your reasoning at any point, correct yourself inside <reflection> tags."
    for pid, item in tqdm(enumerate(data), desc="Submitting tasks", total=len(data)):
        problem = item['problem']
        for id, question in enumerate(item['questions']):
            prompt = get_prompt_system(problem, question['question'], question['options'])
            # prompt = llm.llm_engine.tokenizer.tokenizer.apply_chat_template([{"role":"system", "content":system_prompt}, {"role": "user", "content": prompt}], 
            #                                                                 tokenize=False, add_generation_template=True)
            # print(f"prompt:{prompt}")
            prompts.append(prompt)
            prompt_item_idx.append((pid, id))

    # 计算答案，最多尝试3次
    retry_total = 10
    retry_cnt = 0
    retry_flag = True
    while retry_cnt < retry_total and retry_flag:
        print(f'retry_cnt: {retry_cnt}/{retry_total}')
        retry_flag, prompts, prompt_item_idx = get_outputs(llm, prompts, sampling_params, data, prompt_item_idx, answer_name)
        retry_cnt += 1

    if args.check_model:
        # 检查
        check_answer(data, answer_name)

    else:
        # 补全答案
        fill_answer(data, prompt_item_idx, answer_name)
        
        # 合并文件
        data = merge_problems(data, answer_name)
        print(f'merge_list:{len(data)}')

        # 写文件
        sorted_data = sorted(data, key=lambda x: int(str(x['id'])[-3:]))
        write_file(result_file, sorted_data)
