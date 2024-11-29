export CUDA_VISIBLE_DEVICES=0

MODEL_PATH=ckpts/dpo/llama/iterate_1-merge

OUTPUT=results/llama/code/iterate_1

echo "CKPT on $MODEL_PATH"
echo "Results on $OUTPUT"

if [ ! -d $OUTPUT ];then
    mkdir -p $OUTPUT
fi

python -m eval.codex_humaneval.run_eval \
    --data_file eval/codex_humaneval/HumanEval.jsonl.gz \
    --eval_pass_at_ks 1 5 10 20 \
    --unbiased_sampling_size_n 20 \
    --temperature 0.1 \
    --save_dir $OUTPUT \
    --model $MODEL_PATH \
    --tokenizer $MODEL_PATH \
    --use_vllm \
    --eval_batch_size 1 >> $OUTPUT/eval_process.log
