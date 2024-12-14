#!/bin/bash


name=test
dir_path=data
ckpt=ckpts/Llama-2-7b-hf
judge_model=~/prometheus-7b-v2.0 # you judge model path
split_num=4
gpu_start=$1
offset=$((gpu_start - 1))

tmp_output_dir=tmp
output_dir=data
mkdir $output_dir
mkdir $tmp_output_dir

python preprocess/split_jsonl.py \
    --file_path ${dir_path}/${name}.jsonl \
    --num_splits $split_num \
    --output_dir $tmp_output_dir


for i in $(seq 1 $split_num); do
    CUDA_VISIBLE_DEVICES=$(($i + $offset)) python -m eval.generate.run_eval \
        --data_dir ${tmp_output_dir}/${name}_part${i}.jsonl \
        --save_path ${tmp_output_dir}/generate_result_part${i}.jsonl \
        --model $ckpt \
        --tokenizer $ckpt \
        --eval_batch_size 10 \
        --no_cot \
        --use_vllm &
done

# wait finish 
wait


for i in $(seq 1 $split_num); do
    CUDA_VISIBLE_DEVICES=$(($i + $offset)) python preprocess/score.py \
        --part_name part${i} \
        --model_path $judge_model\
        --input_dir $tmp_output_dir \
        --output_dir $tmp_output_dir &
done

wait

python preprocess/merge_jsonl.py \
    --output_path ${tmp_output_dir}/${name}_all_score.jsonl \
    --merge_file_name score \
    --input_dir ${tmp_output_dir} \
    --num_splits $split_num



python preprocess/selector.py --file_path $tmp_output_dir/${name}_all_score.jsonl --save-dpo $output_dir

rm -rf $tmp_output_dir