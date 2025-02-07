train_data_path=./data/round1_train_llama3.1_sft_self_test.jsonl
rm $train_data_path
touch $train_data_path
echo '[' > $train_data_path
cat ./data/gen_dpo_true.jsonl >> $train_data_path
echo ']' >> $train_data_path
