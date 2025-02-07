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
            if question['answer'] == question[model]:
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
    parser.add_argument('--check_model', action='store_true', help='检查模型效果')
    parser.add_argument('--input_file', type=str, help='输入文件')
    
    args = parser.parse_args()
    print(f'start args: {args}')

    # 设置
    input_file = args.input_file # '/data/root/jupyter/modelscope/round1_train_data.jsonl'

    # 按行读取数据
    data = []
    with open(input_file) as reader:
        for line in reader:
            sample = json.loads(line)
            data.append(sample)

    # 检查
    print(f"{args.check_model}")
    check_answer(data, args.model)