# sft训练
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
llamafactory-cli train examples/train_lora/qwen2.5-32B.yaml

# merge lora
llamafactory-cli export examples/merge_lora/qwen2.5_32b_sft.yaml

# quant
CUDA_VISIBLE_DEVICES=0,1,2,3 \
llamafactory-cli export examples/merge_lora/qwen2.5_32b_quant.yaml