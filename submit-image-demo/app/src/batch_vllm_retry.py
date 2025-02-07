from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from tqdm import tqdm
import argparse
import json
import re
import torch
import gc

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

def get_prompt_reflection_2(problem, question, options):
    options = '\n'.join(f"{'ABCDEFG'[i]}. {o}" for i, o in enumerate(options))
    prompt = f"""你是一个逻辑推理专家，擅长复杂的推理和反思。以下是一个逻辑推理的题目，形式为单项选择题。所有的问题都是（close-world assumption）闭世界假设，即未观测事实都为假。请逐步分析问题，在<思考>标签内描述推理过程，在<反思>标签内检查推理过程是否正确，如果有误及时纠错，在<输出>标签内提供你的最终回答。输出格式为:<输出>因此答案是：A</输出>。题目如下：

        ### 题目:
        {problem}

        ### 问题:
        {question}
        {options}
        """
    return prompt

from collections import Counter
def most_common_element(lst):
    # 使用Counter来统计每个元素的出现次数
    counts = Counter(lst)
    # 返回出现次数最多的元素
    return counts.most_common(1)[0][0]

# 合并答案
def merge_answer(data, answer_name):
    for item in data:
        for id, question in enumerate(item['questions']):
            if answer_name in question:
                if len(question[answer_name]) >= 1:
                    # print(f'{id} {question[answer_name]}')
                    question[answer_name] = most_common_element(question[answer_name])
                    # print(f'final: {question[answer_name]} {question["answer"]}')
                else:
                    question[answer_name] = 'A'

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

def get_outputs(llm, prompts, sampling_params, data, prompt_item_idx, answer_name, lora):
    # 计算结果
    outputs = None
    if lora:
        print(f'generate with lora: {lora}')
        outputs = llm.generate(prompts, sampling_params=sampling_params, 
                               lora_request=LoRARequest("lora_adapter", 1, lora))
    else:
        outputs = llm.generate(prompts, sampling_params=sampling_params, )

    for idx, output in enumerate(outputs):
        res = output.outputs[0].text
        result = extract(res)
        pid, id = prompt_item_idx[idx]
        question = data[pid]['questions'][id]
        if answer_name not in question:
            question[answer_name] = []
        if result != 'NULL':
            question[answer_name].append(result)


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
    parser.add_argument('--model_dir', type=str, default='models', help='model dir')
    parser.add_argument('--lora_dir', type=str, default='lora', help='lora dir')
    parser.add_argument('--lora', type=str, default='', help='lora name')
    parser.add_argument('--answer_name', type=str, help='answer name')
    parser.add_argument('--input_file', type=str, help='输入文件')
    parser.add_argument('--topk', type=int, default=0, help='测试程序是否正常')
    parser.add_argument('--gpu_memory_utilization', type=float, default=1.0, help='gpu使用率')
    parser.add_argument('--tp', type=int, default=1, help='多少卡')
    parser.add_argument('--max_model_len', type=int, default=8192, help='max_model_len')
    parser.add_argument('--check_model', action='store_true', help='检查模型效果')
    parser.add_argument('--retry_total', type=int, default=5, help='重试次数')
    parser.add_argument('--reflection', action='store_true', help='采用reflection')
    parser.add_argument('--temperature', type=float, default=-1, help='temperature')
    parser.add_argument('--top_p', type=float, default=-1, help='top_p')
    parser.add_argument('--quant', type=str, default="", help='quant')
    parser.add_argument('--cpu_offload_gb', type=int, default=0, help="cpu_offload_gb")
    parser.add_argument('--reflection_list', type=str, default='', help='多个模型的配置')
    
    
    
    
    args = parser.parse_args()
    print(f'start args: {args}')

    # 设置
    model_list = args.model.split(',') # 'deepseek-code-v2'
    lora_list = args.lora.split(',')
    if args.reflection_list != '':
        reflection_list = [bool(int(x)) for x in args.reflection_list.split(',')]
    else:
        reflection_list = []
    print(f'relection={args.reflection}, reflection_list={reflection_list}')
  
    input_file = args.input_file # '/data/root/jupyter/modelscope/round1_train_data.jsonl'
    topk = args.topk
    answer_name = args.answer_name

    # 输出文件
    output_dir = f'./data'
    result_file = f'./results.jsonl'
    result_check_file = f'./results_{args.model_dir}_{args.lora_dir}_{args.temperature}_{args.model}.jsonl'

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
    for idx, model in enumerate(model_list):
        llm = None
        lora = None
        if args.lora != '' and idx < len(lora_list):
            lora = f'{args.lora_dir}/{lora_list[idx]}'
            llm = LLM(
                enable_lora=True,
                model=f"./{args.model_dir}/{model}",
                tokenizer=f'./{args.model_dir}/{model}', 
                trust_remote_code=True,
                tensor_parallel_size=tp_size,
                max_model_len=max_model_len,
                enforce_eager=True,
                gpu_memory_utilization=args.gpu_memory_utilization,
                dtype=torch.float16,
            )
        elif args.quant != '':
            llm = LLM(
                model=f"./{args.model_dir}/{model}",
                # tokenizer=f'./{args.model_dir}/{model}',
                quantization=args.quant,
                trust_remote_code=True,
                tensor_parallel_size=tp_size,
                max_model_len=max_model_len,
                enforce_eager=True,
                gpu_memory_utilization=args.gpu_memory_utilization,
                # dtype=torch.float16,
            )
        elif args.cpu_offload_gb > 0:
            llm = LLM(
                model=f"./{args.model_dir}/{model}",
                tokenizer=f'./{args.model_dir}/{model}',
                trust_remote_code=True,
                tensor_parallel_size=tp_size,
                max_model_len=max_model_len,
                enforce_eager=True,
                gpu_memory_utilization=args.gpu_memory_utilization,
                # dtype=torch.float16,
                # cpu_offload_gb=args.cpu_offload_gb,
                max_num_seqs=512,
                max_num_batched_tokens=max_model_len
            )
        else:
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
        prompt_item_idx = []
        for pid, item in tqdm(enumerate(data), desc=f"Submitting {model} tasks", total=len(data)):
            problem = item['problem']
            for id, question in enumerate(item['questions']):
                if args.reflection or (idx < len(reflection_list) and reflection_list[idx]):
                    # print(f'idx={idx}, use reflection')
                    prompt = get_prompt_reflection(problem, question['question'], question['options'])
                else:
                    prompt = get_prompt(problem, question['question'], question['options'])

                prompt = llm.llm_engine.tokenizer.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], 
                                                                                tokenize=False, add_generation_template=True)
                prompts.append(prompt)
                prompt_item_idx.append((pid, id))

        # 计算答案，最多尝试3次
        retry_total = args.retry_total
        retry_cnt = 0
        while retry_cnt < retry_total:
            print(f'retry_cnt: {retry_cnt}/{retry_total}')
            get_outputs(llm, prompts, sampling_params, data, prompt_item_idx, answer_name, lora)
            retry_cnt += 1

        del llm.llm_engine
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        
    if args.check_model:
        # 检查
        write_file(result_check_file, data)
        merge_answer(data, answer_name)
        check_answer(data, answer_name)
        
    else:
        # 写文件
        merge_answer(data, answer_name)
        sorted_data = sorted(data, key=lambda x: int(str(x['id'])[-3:]))
        write_file(result_file, sorted_data)
