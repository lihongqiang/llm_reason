import argparse
import json
import datetime

# 合并原来的问题和已经生成的答案
def get_problems(inf):
    problems = []
    with open(inf, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            problems.append(sample)
    return problems
            
# 合并问题
def merge_problems(data, answer_name):
    problems = {}
    for item in data:
        problem = item['id']
        if problem in problems:
            exist_problem = problems[problem]
            for id, question in enumerate(item['questions']):
                if id <len(exist_problem['questions']) and answer_name not in exist_problem['questions'][id] and id < len(item['questions']) and answer_name in item['questions'][id]:
                    exist_problem['questions'][id][answer_name] = item['questions'][id][answer_name]
        else:
            problems[problem] = item
            
    return problems.values()

def merge_files(base, upload, model):
    base_problems = get_problems(base)
    upload_problems = get_problems(upload)
    return merge_problems(base_problems+upload_problems, model)

def write_file(file, data):
    with open(file, 'w', encoding='utf-8') as writer:
        for sample in data:
            writer.write(json.dumps(sample, ensure_ascii=False))
            writer.write('\n')

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

def get_answer_data(data, answer_name, model):
    for problem in data:
        for question in problem['questions']:
            if model in question:
                question[answer_name] = question[model]
    return data

# 合并原始文件并写入
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='LLM answer build')
    parser.add_argument('--model', type=str, help='model name')
    parser.add_argument('--stage', type=str, help='train/test')
    parser.add_argument('--input_file', type=str, help='输入文件')
    parser.add_argument('--base_file', type=str, help='合并的基础文件')

    args = parser.parse_args()
    print(f'start args: {args}')

    model = args.model # 'deepseek-code-v2'
    stage = args.stage # 'stage'
    input_file = args.input_file
    base_file = args.base_file

    output_dir = f'./data'
    merge_base_path = f'{output_dir}/upload_merge_base.jsonl'
    answer_path = f"./results.jsonl"

    update_problems = merge_files(base_file, input_file, model)
    print(f'merge problems: {len(update_problems)}')
    update_problems = sorted(update_problems, key=lambda x: int(str(x['id'])[-3:]))
    write_file(merge_base_path, update_problems)

    # 找出缺失的序号
    missing_ids, total_id = find_missing_ids(update_problems, model)
    print("缺失的序号:", missing_ids)
    print(f"missing: {len(missing_ids)}/{total_id}")

    # 找出缺失的problem
    missing_pro = find_missing_problems(update_problems)
    print("缺失的问题:", missing_pro)

    # 写最终答案
    answer_data = get_answer_data(update_problems, 'answer', model)
    write_file(answer_path, answer_data)
