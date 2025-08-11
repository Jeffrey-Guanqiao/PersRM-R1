export WANDB_API_KEY=

deepspeed --num_gpus 8 --module openrlhf.cli.train_sft \
    --save_path  \
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
    --train_batch_size 128 \
    --micro_train_batch_size 16 \
    --pretrain  \
    --bf16 \
    --max_epochs 1 \
    --max_len 12288 \
    --zero_stage 2 \
    --learning_rate 5e-6 \
    --dataset  \
    --apply_chat_template \
    --input_key context_messages \
    --output_key winner \
    --flash_attn \
    --gradient_checkpointing \
    --packing_samples \
    --use_wandb your_wandb \
    --wandb_project  \
    --wandb_run_name 