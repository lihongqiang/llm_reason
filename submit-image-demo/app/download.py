# model_download.py
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer

cnt = 0
while True:
    try:
        model_dir = snapshot_download('qwen/Qwen2.5-72B-Instruct', 
                              cache_dir='/data/root/submit-image-demo/app/models', 
                              revision='master', 
                              ignore_file_pattern=['.pth'])
    except Exception as e:
        print(e)
        cnt += 1

    if cnt > 100:
        break
    
#model_dir = snapshot_download('qwen/Qwen2.5-32B-Instruct-GPTQ-Int4', cache_dir='/data/root/submit-image-demo/app/models', revision='master')

