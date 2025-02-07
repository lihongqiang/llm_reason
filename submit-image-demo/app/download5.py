# model_download.py
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer

model_dir = snapshot_download('qwen/Qwen2.5-32B-Instruct-GPTQ-Int4', cache_dir='/data/root/submit-image-demo/app/models', revision='master')
