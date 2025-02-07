CUDA_VISIBLE_DEVICES=0,1,2,3 \
# VLLM_ATTENTION_BACKEND=FLASHINFER \
python3 -m vllm.entrypoints.openai.api_server  \
--model ./models/Qwen2-7B-Instruct  \
--served-model-name Qwen2-7B-Instruct \
--max-model-len=8192 \
--dtype=float16 \
--trust-remote-code \
--enforce-eager \
--gpu_memory_utilization=0.95 \
--tensor-parallel-size=4 \
--max-log-len=10 \
--disable-log-stats \
--port 8000