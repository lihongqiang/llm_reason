from vllm import LLM, SamplingParams
from tqdm import tqdm
import argparse
import json
import re
import torch
import numpy as np
import random
import os

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
    prompt = f"""以下是一个逻辑推理的题目，形式为单项选择题。所有的问题都是（close-world assumption）闭世界假设，即未观测事实都为假。请逐步分析问题选择最正确的答案并在最后一行输出，最后一行的格式为"因此答案是：A"。题目如下：

        ### 题目:
        {problem}

        ### 问题:
        {question}
        {options}
        """
    return prompt

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

def get_build_prompt(problem, question, options, answer):
    options = '\n'.join(f"{'ABCDEFG'[i]}. {o}" for i, o in enumerate(options))
    prompt = f"""你是一个逻辑推理专家，擅长复杂的推理和反思。以下是一个单项选择题，所有的问题都是（close-world assumption）闭世界假设，即未观测事实都为假。根据给出的答案，请逐步分析问题，在<thinking>标签内描述推理过程，在<reflection>标签内检查推理过程是否正确，如果有误及时纠错，在<output>标签内提供你的最终回答。输出格式为:<output>因此答案是：A</output>。题目如下：

        ### 题目:
        {problem}

        ### 问题:
        {question}
        {options}

        ### 答案:
        {answer}
        """
    return prompt

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
    with open(file, 'a', encoding='utf-8') as writer:
        writer.write(json.dumps(data, ensure_ascii=False, indent=4))
        writer.write(',\n')


def get_outputs(llm, prompts, prompts_chat, sampling_params, data, prompt_item_idx, answer_name, data_file):
    # 计算结果
    outputs = llm.generate(prompts_chat, sampling_params=sampling_params)
    left_prompts = []
    left_prompts_chat = []
    left_prompt_item_idx = []
    for idx, output in enumerate(outputs):
        prompt = prompts[idx]
        res = output.outputs[0].text
        result = extract(res)
        pid, id = prompt_item_idx[idx]
        question = data[pid]['questions'][id]
        question_id = f"{data[pid]['id']}_{id}"
        if result != 'NULL' and result == question['answer']:
            result_prompt = {
                "instruction": prompt,
                "input": "",
                "output": res,
                answer_name: result,
                "answer": question['answer'],
                "id": question_id
            }
            write_file(data_file, result_prompt)
        else:
            left_prompts.append(prompt)
            left_prompts_chat.append(prompt_chat)
            left_prompt_item_idx.append((pid, id))
    print(f'generate: {len(outputs)-len(left_prompts)}/{len(outputs)}')
    return len(left_prompts), left_prompts, left_prompts_chat, left_prompt_item_idx

def fill_answer(data, prompt_item_idx, answer_name):
    for pid, id in prompt_item_idx:
        question = data[pid]['questions'][id]
        question[answer_name] = 'A'
        print(f'fill answer: {pid}, {id}, A')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='LLM API exec')
    parser.add_argument('--model', type=str, help='model name')
    parser.add_argument('--model_dir', type=str, default='models', help='model dir')
    parser.add_argument('--input_file', type=str, help='输入文件')
    parser.add_argument('--topk', type=int, default=0, help='测试程序是否正常')
    parser.add_argument('--tp', type=int, default=1, help='多少卡')
    parser.add_argument('--gpu_memory_utilization', type=float, default=1.0, help='gpu使用率')
    parser.add_argument('--retry_total', type=int, default=1, help='尝试多少次')
    parser.add_argument('--max_model_len', type=int, default=8192, help='max_model_len')
    parser.add_argument('--reflection', action='store_true', help='采用reflection')
    parser.add_argument('--temperature', type=float, default=-1, help='temperature')
    parser.add_argument('--top_p', type=float, default=-1, help='top_p')
    parser.add_argument('--build_answer', action='store_true', help='是否根据答案生成推理过程')
    parser.add_argument('--same_break', action='store_true', help='是否没有新结果就退出')
    
    
    
    args = parser.parse_args()
    print(f'start args: {args}')

    # 设置
    model = args.model # 'deepseek-code-v2'
    input_file = args.input_file # '/data/root/jupyter/modelscope/round1_train_data.jsonl'
    topk = args.topk
    answer_name = model

    # 输出文件
    data_file = f'./data/{model}_sft.jsonl'
    error_file = f'./data/{model}_error.jsonl'


    exist_data_set = set()
    if os.path.exists(data_file):
        with open(data_file, 'r') as f:
            lines = f.readlines()
            data = json.loads(' '.join(["["]+lines+["]"]))
            exist_data_set = set([x['id'] for x in data])

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
    llm = LLM(
        model=f"./{args.model_dir}/{model}",
        tokenizer=f'./{args.model_dir}/{model}',
        trust_remote_code=True,
        tensor_parallel_size=tp_size,
        max_model_len=max_model_len,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=torch.float16,
    )
    if args.temperature == -1:
        if args.top_p == -1:
            sampling_params = SamplingParams(max_tokens=max_model_len)
        else:
            sampling_params = SamplingParams(top_p=args.top_p, max_tokens=max_model_len)
    else:
        if args.top_p == -1:
            sampling_params = SamplingParams(temperature=args.temperature, max_tokens=max_model_len)
        else:
            sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=max_model_len)

    # 生成prompt
    prompts = []
    prompts_chat = []
    prompt_item_idx = []
    for pid, item in tqdm(enumerate(data), desc=f"Submitting {model} tasks", total=len(data)):
        problem = item['problem']
        for id, question in enumerate(item['questions']):

            if f'{item["id"]}_{id}' in exist_data_set:
                print(f'{item["id"]}_{id} exists, continue')
                continue
            
            if args.build_answer:
                prompt = get_build_prompt(problem, question['question'], question['options'], question['answer'])
            elif args.reflection:
                prompt = get_prompt_reflection(problem, question['question'], question['options'])
            else:
                prompt = get_prompt(problem, question['question'], question['options'])
            prompt_chat = llm.llm_engine.tokenizer.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], 
                                                                            tokenize=False, add_generation_template=True)
            prompts.append(prompt)
            prompts_chat.append(prompt_chat)
            prompt_item_idx.append((pid, id))

    # 计算答案，最多尝试1次
    retry_total = args.retry_total
    retry_cnt = 0
    retry_flag = True
    results = {}
    pre_left_num = 0
    while retry_cnt < retry_total:
        print(f'build data retry_cnt: {retry_cnt}/{retry_total}')
        left_num, prompts, prompts_chat, prompt_item_idx = get_outputs(llm, prompts, prompts_chat, sampling_params, data, prompt_item_idx, answer_name, data_file)
        if left_num == 0:
            print(f'break, left_num={left_num}')
            break
        elif args.same_break and left_num == pre_left_num:
            print(f'break, left_num={left_num}, pre_left_num={pre_left_num}')
            break
        else:
            retry_cnt += 1
            pre_left_num = left_num

    # 写失败文件
    for idx, prompt in enumerate(prompts):
        prompt = prompts[idx]
        pid, id = prompt_item_idx[idx]
        question = data[pid]['questions'][id]
        question_id = f"{data[pid]['id']}_{id}"
        result_prompt = {
            "instruction": prompt,
            "input": "",
            "output": "",
            answer_name: "",
            "answer": question['answer'],
            "id": question_id
        }
        write_file(error_file, result_prompt)

    # sft_data = []
    # error_data = []
    # for pid, question in results.items():
    #     is_true = (np.array(question[answer_name]) == question['answer'])
    #     true_cnt = sum(is_true)
    #     if true_cnt > 0 and true_cnt < len(is_true):
    #         sft_data.append({
    #             "instruction": question['instruction'],
    #             "input": "",
    #             "output": random.choice(np.array(question['output'])[is_true]),
    #             answer_name: question[answer_name],
    #             "answer": question['answer'],
    #             "id": question['id']
    #         })
    #     elif true_cnt == 0:
    #         error_data.append({
    #             "instruction": question['instruction'],
    #             "input": "",
    #             "output": question['output'],
    #             answer_name: question[answer_name],
    #             "answer": question['answer'],
    #             "id": question['id']
    #         })
    # print(f"sft data: {len(sft_data)}/{len(results)}, error data: {len(error_data)}/{len(results)}")

    # 写文件
    # sft_data = sorted(sft_data, key=lambda x: x['id'])
    # error_data = sorted(error_data, key=lambda x: x['id'])
    # write_file(data_file, sft_data)
    # write_file(error_file, error_data)

# qwen2 sft data: 1247/1415, error data: 29/1415
# llama3.1 sft data: 1294/1415, error data: 10/1415
# llama3.1 sft sft data: 1316/1415, error data: 10/1415
# llama3.1 sft reflection data: 1146/1415, error data: 18/1415
# glm4 sft data: 1160/1415, error data: 24/1415
# gemma2-int8
