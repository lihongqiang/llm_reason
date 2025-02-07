from vllm import LLM, SamplingParams
from tqdm import tqdm
import argparse
import json
import re
import torch
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from self_discover import select_reasoning_modules, adapt_reasoning_modules, \
    implement_reasoning_structure, execute_reasoning_structure, \
    reasoning_modules_en, reasoning_modules_zh

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
    prompt = f"""
        你是一个逻辑推理专家，擅长解决逻辑推理问题。以下是一个逻辑推理的题目，形式为单项选择题。所有的问题都是（close-world assumption）闭世界假设，即未观测事实都为假。请逐步分析问题选择最正确的答案并在最后一行输出，最后一行的格式为"因此答案是：A"。题目如下：

        ### 题目:
        {problem}

        ### 问题:
        {question}
        {options}
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

def check_answer(data, model):
    true_cnt = 0
    total_cnt = 0
    for item in data:
        for question in item['questions']:
            if model in question and question['answer'] == question[model]:
                true_cnt += 1
            total_cnt += 1
    print(f"true_cnt/total={true_cnt}/{total_cnt}")

def write_file(file, data):
    with open(file, 'w', encoding='utf-8') as writer:
        for sample in data:
            writer.write(json.dumps(sample, ensure_ascii=False))
            writer.write('\n')

import requests
from http import HTTPStatus
def call_ollama_api(model_name, query, port=11434):
    data = {
        'model': model_name,
        'prompt':query,
        "stream": False
    }
    url = f'http://localhost:{port}/api/generate'
    # print(f'call ollama api:\ndata={data}\nurl={url}\n')
    response = requests.post(url, json=data)
    if response.status_code == HTTPStatus.OK:
        return response.json()['response']
    else:
        print('Status code: %s' % (
            response.status_code
        ))
        raise Exception()

def call_vllm_api(model_name, query, port, system):
    
    # Set OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"
    openai_api_base = f"http://localhost:{port}/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    chat_response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": query},
        ]
    )
    return chat_response.choices[0].message.content

def call_self_discover(model, prompt, port):
    selected_modules = select_reasoning_modules(call_vllm_api, model, prompt, port, reasoning_modules_zh)
    # print("Stage 1 SELECT: Selected Modules:\n", selected_modules)
    
    adapted_modules = adapt_reasoning_modules(call_vllm_api, model, prompt, port, selected_modules)
    # print("\nStage 1 ADAPT: Adapted Modules:\n", adapted_modules)
    
    reasoning_structure = implement_reasoning_structure(call_vllm_api, model, prompt, port, adapted_modules)
    # print("\nStage 1 IMPLEMENT: Reasoning Structure:\n", reasoning_structure)

    result = execute_reasoning_structure(call_vllm_api, model, prompt, port, reasoning_structure)
    # print("\nStage 2: Final Result:\n", result)
    return result


def get_outputs(data, prompts, prompt_item_idx, model, port, api_type):

    result_cnt = 0
    left_prompts = []
    left_prompt_item_idx = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        future_data = {}
        for prompt, (pid, id) in tqdm(zip(prompts, prompt_item_idx), desc="Submitting tasks", total=len(prompts)):
            if api_type=='discover':
                future = executor.submit(call_self_discover, model, prompt, port)
            elif api_type=='ollama':
                future = executor.submit(call_ollama_api, model, prompt, port)
            else:
                future = executor.submit(call_vllm_api, model, prompt, port)
            future_data[future] = (prompt, pid, id)

        for future in tqdm(as_completed(future_data), desc="Processing tasks",  total=len(prompts)):
            prompt, pid, id = future_data[future]
            try:
                res  = future.result(timeout=30)
                extract_response = extract(res)
                if extract_response != 'NULL':
                    result_cnt += 1
                    question = data[pid]['questions'][id]
                    question[model] = extract_response
                    # print(f"prompt:{prompt}\nresult:{res}\nanswer:{extract_response}")
                else:
                    left_prompts.append(prompt)
                    left_prompt_item_idx.append((pid, id))
                
            except Exception as e:
                print(f"Failed to process text, Error: {e}")
    print(f'result: {result_cnt}/{len(prompts)}')
    return result_cnt!=len(prompts), left_prompts, left_prompt_item_idx

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
    parser.add_argument('--port', type=int, default=8000, help='端口')
    parser.add_argument('--check_model', action='store_true', help='检查模型效果')
    parser.add_argument('--api_type', type=str, choices=['vllm', 'ollama', 'discover'], default='vllm', help='选择api格式')

    
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

    # 计算结果
    # 生成prompt
    prompts = []
    prompt_item_idx = []
    for pid, item in tqdm(enumerate(data), desc="Prompt build tasks", total=len(data)):
        problem = item['problem']
        for id, question in enumerate(item['questions']):
            prompt = get_prompt(problem, question['question'], question['options'])
            prompts.append(prompt)
            prompt_item_idx.append((pid, id))

    # 计算答案，最多尝试3次
    retry_total = 3
    retry_cnt = 0
    retry_flag = True
    while retry_cnt < retry_total and retry_flag:
        print(f'retry_cnt: {retry_cnt}/{retry_total}')
        retry_flag, prompts, prompt_item_idx = get_outputs(data, prompts, prompt_item_idx,\
                                            answer_name, args.port, args.api_type)
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

