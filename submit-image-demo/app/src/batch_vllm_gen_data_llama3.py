
import torch

model = 'Meta-Llama-3.1-8B-Instruct'


import transformers
import torch

pipeline = transformers.pipeline(
    "text-generation",
    model=f"./models/{model}",
    trust_remote_code=True,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages = [
    {"role": "user", "content": "你是谁"},
]

outputs = pipeline(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])