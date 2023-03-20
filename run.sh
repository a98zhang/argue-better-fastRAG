# create PLAID

python3.8 scripts/indexing/create_plaid.py 
    --checkpoint="Intel/ColBERT-NQ" \
    --collection=data/effective/train/collection.tsv \
    --index-save-path=experiments/notebook --gpus=0 \
    --ranks=1 \
    --name=plaid_test \
    --kmeans_iterations=4



python scripts/training/train_fid.py
--do_train \
--do_eval \
--output_dir output_dir \
--train_file path/to/train_file \
--validation_file path/to/validation_file \
--passage_count 100 \
--model_name_or_path t5-base \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--seed 42 \
--gradient_accumulation_steps 8 \
--learning_rate 0.00005 \
--optim adamw_hf \
--lr_scheduler_type linear \
--weight_decay 0.01 \
--max_steps 15000 \
--warmup_step 1000 \
--max_seq_length 250 \
--max_answer_length 20 \
--evaluation_strategy steps \
--eval_steps 2500 \
--eval_accumulation_steps 1 \
--gradient_checkpointing \
--bf16 \
--bf16_full_eval