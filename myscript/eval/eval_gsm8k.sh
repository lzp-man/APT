export CUDA_VISIBLE_DEVICES=0

DATA_DIR=eval/gsm

MODEL_PATH=ckpts/dpo/llama/iterate_1-merge

OUTPUT=results/llama/gsm8k/iterate_1

echo "CKPT on $MODEL_PATH"
echo "results on $OUTPUT"

if [ ! -d $OUTPUT ];then
    mkdir -p $OUTPUT
fi

python -m eval.gsm.run_eval \
    --data_dir $DATA_DIR \
    --save_dir $OUTPUT \
    --model $MODEL_PATH \
    --tokenizer $MODEL_PATH \
    --eval_batch_size 1 \
    --n_shot 8 \
    --use_vllm \

