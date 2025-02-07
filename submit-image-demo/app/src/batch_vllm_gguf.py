from vllm import LLM, SamplingParams
import torch

def run_gguf_inference(model_path, token_path):
    PROMPT_TEMPLATE = "<|start_header_id|>user<|end_header_id|>\n\n{{content}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"  # noqa: E501
    # Sample prompts.
    prompts = [
        "你是一个逻辑推理专家，擅长解决逻辑推理问题。以下是一个逻辑推理的题目，形式为单项选择题。所有的问题都是（close-world assumption）闭世界假设，即未观测事实都为假。请逐步分析问题选择最正确的答案并在最后一行输出，最后一行的格式为\"因此答案是：A\"。题目如下：\n\n        ### 题目:\n        有几种不同的方法来计算数字的幂。在这里，你需要根据不同的计算步骤确定相应的结果。\n\n        ### 问题:\n        选择题 3：\n若数 9 将自己乘 9 次，最终结果是多少？\n        A. 387420489\nB. 3486784401\nC. 43\nD. 59049\n        ",
    ]
    prompts = [
        PROMPT_TEMPLATE.format(prompt=prompt)
        for prompt in prompts
    ]
    # Create a sampling params object.
    sampling_params = SamplingParams(n=1, best_of=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, 
                                     temperature=0.7, top_p=1.0, top_k=-1, min_p=0.0, seed=None, use_beam_search=False, length_penalty=1.0, 
                                     early_stopping=False, stop=[], stop_token_ids=[], include_stop_str_in_output=False, ignore_eos=False, 
                                     max_tokens=None, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, 
                                     spaces_between_special_tokens=True, truncate_prompt_tokens=None)
    
    llm = LLM(model=model_path,
            tokenizer=token_path, 
            gpu_memory_utilization=1.0,
            skip_tokenizer_init=False, tokenizer_mode='auto', 
            revision=None, rope_scaling=None, rope_theta=None, 
            tokenizer_revision=None, 
            trust_remote_code=True, 
            dtype=torch.float16, 
            max_model_len=1024,
            download_dir=None, 
            tensor_parallel_size=1, 
            pipeline_parallel_size=1, 
            disable_custom_all_reduce=False, 
            # quantization='auto', 
            enforce_eager=True, 
            kv_cache_dtype='auto',
            quantization_param_path=None,
            use_v2_block_manager=False, enable_prefix_caching=False)

    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == "__main__":
    run_gguf_inference('/data/root/submit-image-demo/app/models/Meta-Llama-3.1-70B-Instruct-gguf/Meta-Llama-3.1-70B-Instruct-IQ3_XS.gguf', 
                       '/data/root/submit-image-demo/app/models/Meta-Llama-3.1-70B-Instruct-gguf')