# 批量跑结果
import torch
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import time
from tqdm import tqdm
from openai import OpenAI
import argparse


def get_available_gpu():
    num_gpu = torch.cuda.device_count()
    if num_gpu > 0:
        return [i for i in range(num_gpu)]
    else:
        return None
 
def get_prompt(problem, question, options, prompt_template, mode, answer, try_true):

    options = '\n'.join(f"{'ABCDEFG'[i]}. {o}" for i, o in enumerate(options))
    if mode in ['infer', 'gen_dpo']:
        if prompt_template == 1:
            prompt = f"""你是一个逻辑推理专家，擅长解决逻辑推理问题。以下是一个逻辑推理的题目，形式为单项选择题。所有的问题都是（close-world assumption）闭世界假设，即未观测事实都为假。请逐步分析问题选择最正确的答案并在最后一行输出，最后一行的格式为"因此答案是：A"。题目如下：

        ### 题目:
        {problem}

        ### 问题:
        {question}
        {options}
        """
        elif prompt_template == 2:
            prompt = f"""CONTEXT（上下文）
    你是一位专业的逻辑推理专家，面对的是一个闭世界假设下的逻辑推理题目。闭世界假设意味着所有未被明确陈述的事实都视为不真实。你的目标是通过逐步分析，使用逻辑和推理来解决这个问题。

    OBJECTIVE（目标）
    分析给定的逻辑推理题目，利用闭世界假设，识别并选择最合理的答案。你的任务是提供一个清晰、逻辑上一致的解答过程。

    STYLE（风格）
    采用专业、条理清晰、简单概要的风格，确保推理过程易于理解和跟随。使用逻辑推理领域内的标准术语和表达方式。

    TONE（语调）
    保持客观和中立，专注于逻辑推理的准确性和有效性，避免情感色彩的干扰。

    AUDIENCE（受众）
    目标受众是对逻辑推理感兴趣的专业人士或学生，他们寻求深入分析和精确解答。

    RESPONSE（响应）
    输出应包括推理的每个步骤和最终答案。确保最终答案以明确的格式呈现。

    INSTRUCTIONS（指令）
    仔细阅读题目和选项。
    根据闭世界假设，排除所有不可能的选项。
    使用逻辑推理，逐步分析剩余选项。
    确定最合理的答案，并在最后明确地输出。

    FINAL OUTPUT FORMAT（最终输出格式）
    你的最终输出必须严格遵循以下格式：
    “答案是：[正确选项]#end”
    示例正确输出：经过逻辑推理，结论是：答案是：B #end
    示例正确输出：答案是：B 5#end
    示例正确输出：答案是：B 否#end
    示例正确输出：答案是：B [1,2,3]#end
    示例错误输出：可能的答案是B，但我也不确定... #end
    示例错误输出：通过上述分析，只有选项B符合程序的行为描述和逻辑推理#end。
    示例错误输出：只有选项C符合条件。

    ### 题目:
    {problem}

    ### 问题:
    {question}
    {options}
    """
        elif prompt_template == 3:
            prompt = f"""你是一位专业的逻辑推理专家，面对的是一个闭世界假设下的逻辑推理题目，需要通过逐步分析来解决，保证逻辑推理的准确性。
    使用逻辑推理，仔细分析题目和选项，排除不可能的选项，选择最合理的选项，按照响应格式输出最终答案。
    最终答案的格式为“答案是：[正确选项]”

    ### 题目:
    {problem}

    ### 问题:
    {question}
    {options}
    """
        elif prompt_template == 4:
            prompt = f"""你是一个逻辑推理专家，擅长解决逻辑推理问题。以下是一个逻辑推理的题目，形式为单项选择题。所有的问题都是（close-world assumption）闭世界假设，即未观测事实都为假。请逐步分析问题，推理过程严谨，理性和客观，选择最正确的答案并在最后一行输出，最后一行的格式为"因此答案是：A"。题目如下：

        ### 题目:
        {problem}

        ### 问题:
        {question}
        {options}
        """
        elif prompt_template == 5:
            prompt = f"""你是一个逻辑推理专家，擅长解决逻辑推理问题。以下是一个逻辑推理的题目，形式为单项选择题。所有的问题都是（close-world assumption）闭世界假设，即未观测事实都为假。请逐步分析问题选择最正确的答案并在最后一行输出，最后一行的格式为"因此答案是：A"，其中A是正确选项。题目如下：

        ### 题目:
        {problem}

        ### 问题:
        {question}
        {options}
        """
        else:
            prompt = ''
    elif mode in ['gen_data']:
        prompt = f"""你是一个逻辑推理专家，擅长解决逻辑推理问题。以下是一个逻辑推理的题目，形式为单项选择题。所有的问题都是（close-world assumption）闭世界假设，即未观测事实都为假。请根据答案逐步分析问题给出推理过程，并在最后一行输出答案，最后一行的格式为"因此答案是：A"。题目如下：

        ### 题目:
        {problem}

        ### 问题:
        {question}
        {options}

        ### 答案:
        {answer}
        """
    else:
        print('mode error.')
    return prompt

# 这里使用extract抽取模获得抽取的结果，如果不在可选范围内，选最后一个
import re
def extract(input_text, options=['A', 'B', 'C', 'D', 'E', 'F', 'G']):
    try:
        problems = re.findall(r"答案是[:： ,]*([A-G])", input_text.strip('\n'))
        print(f"problems: {problems}")
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

def find_missing_ids(data, model):
    total_id = 0
    missing_ids = set()
    for item in data:
        for id, question in enumerate(item['questions']):
            if model not in question:
                missing_ids.add(f"{item['id']}_{id}")
            total_id += 1
    return sorted(missing_ids), total_id
                

def find_missing_problems(dict_list):
    # 提取所有序号
    extracted_ids = {int(d['id'][-3:]) for d in dict_list}
    
    # 创建0-500的序号集合
    all_ids = set(range(500))
    
    # 找出缺失的序号
    missing_ids = all_ids - extracted_ids
    
    return sorted(missing_ids)

# 合并原来的问题和已经生成的答案
def get_problems(inf):
    problems = []
    with open(inf, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            problems.append(sample)
    return problems
            
    
def merge_files(base, upload, model):
    base_problems = get_problems(base)
    upload_problems = get_problems(upload)
    print(f'base_problems: {len(base_problems)}, upload_robems: {len(upload_problems)}')
    return merge_problems(base_problems+upload_problems, model)

def write_file(file, data):
    with open(file, 'w', encoding='utf-8') as writer:
        for sample in data:
            writer.write(json.dumps(sample, ensure_ascii=False))
            writer.write('\n')

def evaluate(data, base_name, model, error_file):
    pse = 0
    cnt = 0
    tot = 0
    for task in data:
        for question in task['questions']:
            if model in question:
                
                tot += 1
                cnt += question[base_name] == question[model]
            else:
                pse += 1
    ans = f'cnt={cnt}/{tot}, score={cnt/tot}, total cnt={cnt}/{(tot+pse)}, total score={cnt/(tot+pse)}'
    print(ans)
    with open(error_file, 'w', encoding='utf-8') as writer:
        writer.write(f'{ans}\n')

import requests
from http import HTTPStatus
def call_ollama_api(model_name, query, port):
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

def call_vllm_api(model_name, query, port):
    
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

def api_retry(MODEL_NAME, query, port, is_ollama, mode, answer, try_true):
    max_retries = 6
    retry_delay = 60  # in seconds
    attempts = 0
    while attempts < max_retries:
        try:
            if is_ollama:
                res = call_ollama_api(MODEL_NAME, query, port)
            else:
                res = call_vllm_api(MODEL_NAME, query, port)
            extract_res = extract(res)
            if (mode in ['infer'] and extract_res == 'NULL') or (mode in ['gen_data', 'gen_dpo'] and try_true and extract_res != answer):
                attempts += 1
                print(f"mode={mode}, Attempt {attempts} failed for text: {query}. res: {res}, extract_res: {extract_res}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                return res
        except Exception as e:
            attempts += 1   
            if attempts < max_retries:
                print(f"Attempt {attempts} failed for text: {query}. Retrying in {retry_delay} seconds...\nError: {e}")
                time.sleep(retry_delay)
            else:
                print(f"All {max_retries} attempts failed for text: {query}. Error: {e}")
                raise

def process_datas(datas, model, ofn, error_file, port, prompt_template, is_ollama, mode, gen_file, try_true):
    # 记录中间结果
    writer = open(ofn, 'w', encoding='utf-8')
    error_write = open(error_file, 'w', encoding='utf-8')
    gen_write = open(gen_file, 'a')
    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_data = {}
        lens = 0
        for data in tqdm(datas, desc="Submitting tasks", total=len(datas)):
            problem = data['problem']
            for id, question in enumerate(data['questions']):
                
                # 如果已经有答案就不在请求api
                if model in question:
                    if mode in ['gen_dpo', 'gen_data'] and try_true:
                        if model in question and question[model] == question['answer']:
                            results.append(data)
                            writer.write(f'{json.dumps(data, ensure_ascii=False)}\n')
                            print(f"answer exists and is write: {data['id']}_{id}")
                            continue
                    else:
                        results.append(data)
                        writer.write(f'{json.dumps(data, ensure_ascii=False)}\n')
                        print(f"answer exists: {data['id']}_{id}")
                        continue

                if mode in ['gen_data', 'gen_dpo']:
                    answer = question['answer']
                else:
                    answer = ''
                prompt = get_prompt(problem, question['question'], question['options'], prompt_template, mode, answer, try_true)
                future = executor.submit(api_retry, model, prompt, port, is_ollama, mode, answer, try_true)
                future_data[future] = (data, id, prompt)
                lens += 1
                
        for future in tqdm(as_completed(future_data), total=lens, desc="Processing tasks"):

            data = future_data[future][0]
            problem_id = future_data[future][1]
            prompt = future_data[future][2]
            try:
                res  = future.result()
                extract_response = extract(res)
                if extract_response != 'NULL':
                    question = data['questions'][problem_id]
                    question[model] = extract_response

                    if mode == 'gen_data':
                        ori_prompt = get_prompt(data['problem'], question['question'], question['options'], \
                                                prompt_template, 'infer', '')
                        gen_data = {
                            "instruction": ori_prompt,
                            "input": "",
                            "output": res
                        }
                        json.dump(gen_data, gen_write, ensure_ascii=False, indent=4)
                        gen_write.write(',\n')
                    elif mode == 'gen_dpo':
                        gen_data = {
                            "instruction": prompt,
                            "input": "",
                            "output": res,
                            "model": extract_response,
                            "answer": question['answer'],
                            "id": f"{data['id']}_{problem_id}"
                        }
                        json.dump(gen_data, gen_write, ensure_ascii=False, indent=4)
                        gen_write.write(',\n')
                else:
                    error_dict = {
                        'prompt':prompt,
                        'output':res,
                        'answer':extract_response
                    }
                    error_write.write(json.dumps(error_dict))


                    if mode == 'gen_dpo':
                        question = data['questions'][problem_id]
                        question[model] = extract_response
                        gen_data = {
                            "instruction": prompt,
                            "input": "",
                            "output": res,
                            "model": extract_response,
                            "answer": question['answer'],
                            "id": f"{data['id']}_{problem_id}"
                        }
                        json.dump(gen_data, gen_write, ensure_ascii=False, indent=4)
                        gen_write.write(',\n')
                    elif mode == 'infer' and try_true:
                        print(f"mode={mode}, try_true={try_true}, answer is NULL, change to A.")
                        question = data['questions'][problem_id]
                        question[model] = 'A'
                results.append(data)
                writer.write(f'{json.dumps(data, ensure_ascii=False)}\n')
                
            except Exception as e:
                print(f"Failed to process text: {data}. Error: {e}")
    writer.close()
    error_write.close()
    gen_write.close()
    return results

def main(ifn, ofn, model, error_file, topk, port, prompt_template, is_ollama, mode, gen_file, try_true):
    data = []
    # 按行读取数据
    with open(ifn) as reader:
        for line in reader:
            sample = json.loads(line)
            data.append(sample)
    if topk>0:
        data = data[:topk]
    data_list = process_datas(data,model,ofn,error_file, port, prompt_template, is_ollama, mode, gen_file, try_true)
    print(f"All tasks finished! data_list:{len(data_list)}")
    return data_list

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='LLM API exec')
    parser.add_argument('--model', type=str, help='model name')
    parser.add_argument('--stage', type=str, help='train/test/gen_dpo')
    parser.add_argument('--input_file', type=str, help='输入文件')
    parser.add_argument('--eval', action='store_true', default=False, help='是否评估')
    parser.add_argument('--is_ollama', action='store_true', help='是否调用ollama api')
    parser.add_argument('--topk', type=int, default=0, help='测试程序是否正常')
    parser.add_argument('--port', type=int, default=8000, help='服务端口')
    parser.add_argument('--prompt_template', type=int, default=1, choices=[1,2,3,4,5], help='prompt模版')
    parser.add_argument('--mode', type=str, choices=['infer', 'gen_data', 'gen_dpo'], default='infer', help='infer/gen_data/gen_dpo')
    parser.add_argument('--try_true', action='store_true', default=False, help='是否尽量保证答案正确')
    

    args = parser.parse_args()
    print(f'start args: {args}')

    # 设置
    model = args.model # 'deepseek-code-v2'
    stage = args.stage # 'stage'
    input_file = args.input_file # '/data/root/jupyter/modelscope/round1_train_data.jsonl'
    topk = args.topk
    port = args.port
    is_eval = args.eval
    is_ollama = args.is_ollama
    prompt_template = args.prompt_template
    mode = args.mode
    try_true = args.try_true

    # 输出文件
    output_dir = f'./data'
    upload_ifle = f'{output_dir}/upload.jsonl'
    merge_file = f'{output_dir}/upload_merge.jsonl'
    error_file = f'{output_dir}/error.jsonl'
    if mode in ['gen_data', 'gen_dpo']:
        if try_true:
            gen_file = f'{output_dir}/{mode}_true.jsonl'
        else:
            gen_file = f'{output_dir}/{mode}.jsonl'
    else:
        gen_file = 'gen_file.jsonl'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 初始模式
    data_list = main(input_file, upload_ifle, model, error_file, topk, port, prompt_template, is_ollama, mode, gen_file, try_true)
    # 重跑模式
    # data_list = main(merge_base_path, upload_ifle)

    # 合并文件
    data = merge_problems(data_list, model)
    print(f'merge_list:{len(data)}')
    sorted_data = sorted(data, key=lambda x: int(str(x['id'])[-3:]))
    write_file(merge_file, sorted_data)

    # 训练数据评估
    if is_eval:
        evaluate(sorted_data, 'answer', model, error_file)

    # 找出缺失的序号
    missing_ids, total_id = find_missing_ids(sorted_data, model)
    print("缺失的序号:", missing_ids)
    print(f"missing: {len(missing_ids)}/{total_id}")

    # 找出确实的problem
    missing_pro = find_missing_problems(sorted_data)
    print("缺失的问题:", missing_pro)