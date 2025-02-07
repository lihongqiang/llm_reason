import openai
import os
import time

reasoning_modules_en = [
    "1. How could I devise an experiment to help solve that problem?",
    "2. Make a list of ideas for solving this problem, and apply them one by one to the problem to see if any progress can be made.",
    "3. How could I measure progress on this problem?",
    "4. How can I simplify the problem so that it is easier to solve?",
    "5. What are the key assumptions underlying this problem?",
    "6. What are the potential risks and drawbacks of each solution?",
    "7. What are the alternative perspectives or viewpoints on this problem?",
    "8. What are the long-term implications of this problem and its solutions?",
    "9. How can I break down this problem into smaller, more manageable parts?",
    "10. Critical Thinking: This style involves analyzing the problem from different perspectives, questioning assumptions, and evaluating the evidence or information available. It focuses on logical reasoning, evidence-based decision-making, and identifying potential biases or flaws in thinking.",
    "11. Try creative thinking, generate innovative and out-of-the-box ideas to solve the problem. Explore unconventional solutions, thinking beyond traditional boundaries, and encouraging imagination and originality.",
    "12. Seek input and collaboration from others to solve the problem. Emphasize teamwork, open communication, and leveraging the diverse perspectives and expertise of a group to come up with effective solutions.",
    "13. Use systems thinking: Consider the problem as part of a larger system and understanding the interconnectedness of various elements. Focuses on identifying the underlying causes, feedback loops, and interdependencies that influence the problem, and developing holistic solutions that address the system as a whole.",
    "14. Use Risk Analysis: Evaluate potential risks, uncertainties, and tradeoffs associated with different solutions or approaches to a problem. Emphasize assessing the potential consequences and likelihood of success or failure, and making informed decisions based on a balanced analysis of risks and benefits.",
    "15. Use Reflective Thinking: Step back from the problem, take the time for introspection and self-reflection. Examine personal biases, assumptions, and mental models that may influence problem-solving, and being open to learning from past experiences to improve future approaches.",
    "16. What is the core issue or problem that needs to be addressed?",
    "17. What are the underlying causes or factors contributing to the problem?",
    "18. Are there any potential solutions or strategies that have been tried before? If yes, what were the outcomes and lessons learned?",
    "19. What are the potential obstacles or challenges that might arise in solving this problem?",
    "20. Are there any relevant data or information that can provide insights into the problem? If yes, what data sources are available, and how can they be analyzed?",
    "21. Are there any stakeholders or individuals who are directly affected by the problem? What are their perspectives and needs?",
    "22. What resources (financial, human, technological, etc.) are needed to tackle the problem effectively?",
    "23. How can progress or success in solving the problem be measured or evaluated?",
    "24. What indicators or metrics can be used?",
    "25. Is the problem a technical or practical one that requires a specific expertise or skill set? Or is it more of a conceptual or theoretical problem?",
    "26. Does the problem involve a physical constraint, such as limited resources, infrastructure, or space?",
    "27. Is the problem related to human behavior, such as a social, cultural, or psychological issue?",
    "28. Does the problem involve decision-making or planning, where choices need to be made under uncertainty or with competing objectives?",
    "29. Is the problem an analytical one that requires data analysis, modeling, or optimization techniques?",
    "30. Is the problem a design challenge that requires creative solutions and innovation?",
    "31. Does the problem require addressing systemic or structural issues rather than just individual instances?",
    "32. Is the problem time-sensitive or urgent, requiring immediate attention and action?",
    "33. What kinds of solution typically are produced for this kind of problem specification?",
    "34. Given the problem specification and the current best solution, have a guess about other possible solutions."
    "35. Let's imagine the current best solution is totally wrong, what other ways are there to think about the problem specification?"
    "36. What is the best way to modify this current best solution, given what you know about these kinds of problem specification?"
    "37. Ignoring the current best solution, create an entirely new solution to the problem."
    "38. Let's think step by step."
    "39. Let's make a step by step plan and implement it with good notation and explanation."
]

reasoning_modules_zh = [
    "1. 我该如何设计一个实验来解决这个问题?",
    "2. 列出解决这个问题的想法,然后逐一应用它们来解决这个问题,看看是否可以取得任何进展。", 
    "3. 我怎样才能衡量这个问题的进展？", 
    "4. 我怎样才能简化这个问题以便于解决？", 
    "5. 这个问题背后的关键假设是什么？", 
    "6. 每种解决方案的潜在风险和缺点是什么？", 
    "7. 对于这个问题,还有哪些其他的观点或看法？", 
    "8. 这个问题及其解决方案的长期影响是什么？", 
    "9. 我怎样才能将这个问题分解成更小、更易于管理的部分？", 
    "10. 批判性思维：这种思维方式包括从不同角度分析问题、质疑假设以及评估现有证据或信息。它侧重于逻辑推理、基于证据的决策以及识别思维中的潜在偏见或缺陷。", 
    "11. 尝试创造性思维,产生创新和突破常规的想法来解决问题。探索非常规解决方案,超越传统界限思考,鼓励想象力和独创性。", 
    "12. 寻求他人的意见和合作来解决问题。强调团队合作、开放的沟通,并利用团队的不同观点和专业知识来提出有效的解决方案。", 
    "13. 使用系统思维：将问题视为更大系统的一部分,并了解各种元素之间的相互联系。重点是确定影响问题的根本原因、反馈回路和相互依赖关系,并制定解决整个系统的整体解决方案。", 
    "14. 使用风险分析：评估与问题的不同解决方案或方法相关的潜在风险、不确定性和权衡。强调评估成功或失败的潜在后果和可能性,并根据风险和收益的平衡分析做出明智的决策。", 
    "15. 使用反思性思维：远离问题,花时间进行内省和自我反省。检查可能影响解决问题的个人偏见、假设和思维模式,并乐于从过去的经验中学习,以改进未来的方法。", 
    "16. 需要解决的核心问题是什么？", 
    "17. 导致该问题的根本原因或因素是什么？",
    "18. 是否有任何潜在的解决方案或策略曾经被尝试过？如果有,结果和经验教训是什么？", 
    "19. 解决这个问题可能出现哪些潜在障碍或挑战？", 
    "20. 是否有相关数据或信息可以洞悉问题？如果有,哪些数据源可用,如何分析这些数据？",
    "21. 是否有任何利益相关者或个人直接受到该问题的影响？他们的观点和需求是什么？", 
    "22. 需要哪些资源（财力、人力、技术等）来有效解决该问题？", 
    "23. 如何衡量或评估解决问题的进展或成功？", 
    "24. 可以使用什么指标或度量？", 
    "25. 这个问题是技术问题还是实践问题,需要特定的专业知识或技能？还是更多的是概念问题或理论问题？", 
    "26. 该问题是否涉及物理限制,例如资源、基础设施或空间有限？", 
    "27. 该问题是否与人类行为有关,例如社会、文化或心理问题？", 
    "28. 问题是否涉及决策或计划,需要在不确定或相互竞争的目标下做出选择？", 
    "29. 该问题是否是需要数据分析、建模或优化技术的分析问题？", 
    "30. 这个问题是否是一个需要创造性解决方案和创新的设计挑战？", 
    "31. 这个问题是否需要解决系统性或结构性问题,而不仅仅是个别问题？", 
    "32. 该问题是否具有时效性或紧急性,需要立即关注和采取行动？", 
    "33. 针对这种问题规范通常会产生什么样的解决方案？", 
    "34. 给定问题规范和当前最佳解决方案,猜测其他可能的解决方案。" 
    "35. 让我们想象一下当前的最佳解决方案是完全错误的,还有哪些其他方式来思考问题规范？",
    "36. 鉴于您对此类问题规范的了解,修改当前最佳解决方案的最佳方法是什么？",
    "37. 忽略当前的最佳解决方案,为问题创建一个全新的解决方案。",
    "38. 让我一步步思考",
    "39. 让我们制定一个逐步的计划,并用良好的注释和解释来实施它。"]

# STAGE 1
def select_reasoning_modules(query_openai, model, task_description, port, reasoning_modules):
    """
    Step 1: SELECT relevant reasoning modules for the task.
    """
    prompt = f"Given the task: {task_description}, which of the following reasoning modules are relevant? Do not elaborate on why.\n\n" + "\n".join(reasoning_modules)
    selected_modules = query_openai(model, prompt, port)
    return selected_modules

# STAGE 2
def adapt_reasoning_modules(query_openai, model, task_example, port, selected_modules):
    """
    Step 2: ADAPT the selected reasoning modules to be more specific to the task.
    """
    prompt = f"Without working out the full solution, adapt the following reasoning modules to be specific to our task:\n{selected_modules}\n\nOur task:\n{task_example}"
    adapted_modules = query_openai(model, prompt, port)
    return adapted_modules

# STAGE 3
def implement_reasoning_structure(query_openai, model, task_description, port, adapted_modules):
    """
    Step 3: IMPLEMENT the adapted reasoning modules into an actionable reasoning structure.
    """
    prompt = f"Without working out the full solution, create an actionable reasoning structure for the task using these adapted reasoning modules:\n{adapted_modules}\n\nTask Description:\n{task_description}"
    reasoning_structure = query_openai(model, prompt, port)
    return reasoning_structure

# STAGE 4
def execute_reasoning_structure(query_openai, model, task_instance, port, reasoning_structure):
    """
    Execute the reasoning structure to solve a specific task instance.
    """
    prompt = f"Using the following reasoning structure: {reasoning_structure}\n\nSolve this task, providing your final answer: {task_instance}"
    solution = query_openai(model, prompt, port)
    return solution

# Example usage
if __name__ == "__main__":
    

    task_example = "Lisa has 10 apples. She gives 3 apples to her friend and then buys 5 more apples from the store. How many apples does Lisa have now?"

    selected_modules = select_reasoning_modules(task_example, reasoning_modules_en)
    print("Stage 1 SELECT: Selected Modules:\n", selected_modules)
    
    adapted_modules = adapt_reasoning_modules(selected_modules, task_example)
    print("\nStage 1 ADAPT: Adapted Modules:\n", adapted_modules)
    
    reasoning_structure = implement_reasoning_structure(adapted_modules, task_example)
    print("\nStage 1 IMPLEMENT: Reasoning Structure:\n", reasoning_structure)

    result = execute_reasoning_structure(reasoning_structure, task_example)
    print("\nStage 2: Final Result:\n", result)