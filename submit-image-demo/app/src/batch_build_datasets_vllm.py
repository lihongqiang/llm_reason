from vllm import LLM, SamplingParams
from tqdm import tqdm
import argparse
import json
import re
import torch
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
    
def get_prompt_system(problem, question, options):
    options = '\n'.join(f"{'ABCDEFG'[i]}. {o}" for i, o in enumerate(options))
    system_prompt = "你是一个世界级的人工智能系统，能够进行复杂的推理和反思。在<thinking>标签内推理查询，然后在<output>标签内提供你的最终回答。如果你在任何时候发现你的推理有错误，请在<reflection>标签内纠正自己。"
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

def write_prompt(file, data):
    with open(file, 'a', encoding='utf-8') as writer:
        writer.write(json.dumps(data, ensure_ascii=False, indent=4))


def get_outputs(llm, data, prompt_chats, prompts, prompt_item_idx, model, prompt_instruct):

    result_cnt = 0
    left_prompt_chats = []
    left_prompts =[]
    left_prompt_item_idx = []

    outputs = llm.generate(prompt_chats, sampling_params=sampling_params, )
    for idx, output in enumerate(outputs):
        res = output.outputs[0].text
        result = extract(res)

        pid, id = prompt_item_idx[idx]
        question = data[pid]['questions'][id]
        prompt_chat = prompt_chats[idx]
        prompt = prompts[idx]
        # print(f'prompt:{prompts[idx]}\res:{res}\nresult:{result}\nanswer:{question["answer"]}')
        if result != 'NULL' and result == question['answer']:
            result_cnt += 1
            question[model] = result
            prompt_instruct.append({
                "instruction": prompt,
                "input": "",
                "output": res,
                model: question[model],
                "answer": question['answer'],
                "id": f"{data[pid]['id']}_{id}"
            })
        else:
            left_prompts.append(prompt)
            left_prompt_chats.append(prompt_chat)
            left_prompt_item_idx.append((pid, id))
            

    print(f'result: {result_cnt}/{len(prompts)}')
    return result_cnt!=len(prompts), left_prompt_chats, left_prompts, left_prompt_item_idx

def fill_answer(data, prompt_item_idx, answer_name):
    for pid, id in prompt_item_idx:
        question = data[pid]['questions'][id]
        question[answer_name] = 'A'
        print(f'fill answer: {pid}, {id}, A')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='LLM API exec')
    parser.add_argument('--model', type=str, help='model name')
    parser.add_argument('--answer_name', type=str, default='', help='answer name')
    parser.add_argument('--out_tag', type=str, help='out tag')
    parser.add_argument('--input_file', type=str, help='输入文件')
    parser.add_argument('--topk', type=int, default=0, help='测试程序是否正常')
    parser.add_argument('--reflection', action='store_true', help='采用reflection')
    parser.add_argument('--retry_total', type=int, default=5, help='重试次数')
    parser.add_argument('--tp', type=int, default=1, help='多少卡')
    parser.add_argument('--max_model_len', type=int, default=8192, help='max_model_len')
    parser.add_argument('--gpu_memory_utilization', type=float, default=1.0, help='gpu使用率')


    
    args = parser.parse_args()
    print(f'start args: {args}')

    # 设置
    model = args.model # 'deepseek-code-v2'
    input_file = args.input_file # '/data/root/jupyter/modelscope/round1_train_data.jsonl'
    topk = args.topk
    answer_name = args.answer_name

    # 输出文件
    output_dir = f'./data'
    result_file = f'./data/{model}_sft_{args.out_tag}.jsonl'

    exists_set = set()
    if os.path.exists(result_file):
        with open(result_file, 'r', encoding='utf-8') as f:
            exists_data = json.load(f)
            exists_set = set([x['id'] for x in exists_data])

    # 按行读取数据
    data = []
    with open(input_file) as reader:
        for line in reader:
            sample = json.loads(line)
            data.append(sample)
    if topk>0:
        data = data[:topk]

    llm = LLM(
        model=f"./models/{model}",
        tokenizer=f'./models/{model}', 
        trust_remote_code=True,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=torch.float16,
    )
    sampling_params = SamplingParams(temperature=0, max_tokens=args.max_model_len)
    
    # 计算结果
    # 生成prompt
    prompts = []
    prompt_chats = []
    prompt_item_idx = []
    for pid, item in tqdm(enumerate(data), desc="Prompt build tasks", total=len(data)):
        problem = item['problem']
        for id, question in enumerate(item['questions']):

            idx = f"{item['id']}_{id}"
            if idx in exists_set:
                print(f'{idx} exists, continue.')
                continue

            if args.reflection:
                prompt = get_prompt_reflection(problem, question['question'], question['options'])
            else:
                prompt = get_prompt(problem, question['question'], question['options'])
            prompt_chat = llm.llm_engine.tokenizer.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], 
                                                                                tokenize=False, add_generation_template=True)
            prompts.append(prompt)
            prompt_chats.append(prompt_chat)
            prompt_item_idx.append((pid, id))

    # 计算答案，最多尝试3次
    retry_total = args.retry_total
    retry_cnt = 0
    retry_flag = True
    prompt_instruct = []
    prompt_total = len(prompts)
    while retry_cnt < retry_total and retry_flag:
        print(f'retry_cnt: {retry_cnt}/{retry_total}')
        retry_flag, prompt_chats, prompts, prompt_item_idx = get_outputs(llm, data, prompt_chats, prompts, prompt_item_idx,\
                                            answer_name, prompt_instruct)
        retry_cnt += 1

    # 写prompt_instruct
    print(f'prompt_instruct/prompt_total: {len(prompt_instruct)}/{prompt_total}')
    write_prompt(result_file, prompt_instruct)
