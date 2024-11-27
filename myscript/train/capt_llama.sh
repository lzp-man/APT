# you need 8 GPUs for full finetuning
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=32
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training model using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"


MODEL_PATH=base_model
train_file=data/dpo/iterate_0.jsonl  # retrieval_result data
OUTPUT_PATH=ckpts/dpo/iterate_1   
MERGE_OUTPUT_PATH=$OUTPUT_PATH-merge

if [ ! -d $OUTPUT_PATH ];then
    mkdir -p $OUTPUT_PATH
fi
export MASTER_PORT=29801

# Lora training
accelerate launch \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    open_instruct/dpo_tune_capt.py \
    --model_name_or_path $MODEL_PATH \
    --use_lora \
    --use_reg_dpo_loss \
    --use_sft_reg \
    --use_flash_attn \
    --lora_rank 64 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --tokenizer_name $MODEL_PATH \
    --use_slow_tokenizer \
    --train_file $train_file \
    --max_seq_length 2048 \
    --preprocessing_num_workers 128 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 1e-6 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 5 \
    --output_dir $OUTPUT_PATH \
    --with_tracking \
    --report_to tensorboard \
    --checkpointing_steps 2000 \
    --torch_dtype bfloat16 \
    --seed 42 \
    --logging_steps 1  >> $OUTPUT_PATH/train.log 2>&1 


python open_instruct/merge_lora.py \
    --base_model_name_or_path $MODEL_PATH \
    --lora_model_name_or_path $OUTPUT_PATH \
    --output_dir $MERGE_OUTPUT_PATH \
    --save_tokenizer

mv $OUTPUT_PATH $MERGE_OUTPUT_PATH
