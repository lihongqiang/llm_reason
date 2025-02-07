import json

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

file_list = ["./data/round1_test_data.jsonl", "./data/round1_train_data.jsonl"]
text_list = []
out_file = 'LLaMA-Factory/data/round1_text.json'
for file in file_list:
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            problem = json.loads(line)
            for question in problem['questions']:
                text_list.append({
                    "text": get_prompt(problem['problem'], question['question'], question['options'], 1, 'infer', '', False)
                })
print(len(text_list))
with open(out_file, 'w', encoding='utf-8') as writer:
    json.dump(text_list, writer, ensure_ascii=False, indent=4)