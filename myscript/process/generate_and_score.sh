#!/bin/bash


name=your_input_file_name
dir_path=input_file_dirf_path
ckpt=model_for_generation_and_score

split_num=4
gpu_start=$1
offset=$((gpu_start - 1))

tmp_output_dir=tmp_file_save_path
output_dir=final_result

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

# 等待所有后台任务完成
wait

# 并行执行 score.py 脚本
for i in $(seq 1 $split_num); do
    CUDA_VISIBLE_DEVICES=$(($i + $offset)) python preprocess/score.py \
        --part_name part${i} \
        --input_dir $tmp_output_dir \
        --output_dir $tmp_output_dir &
done

wait

python preprocess/merge_jsonl.py \
    --output_path ${tmp_output_dir}/${name}_all_score.jsonl \
    --merge_file_name score \
    --input_dir ${tmp_output_dir} \
    --num_splits $split_num


python preprocess/selector.py --name $tmp_output_dir/${name}_all_score.jsonl --save-dpo $output_dir

# you will get error_case.jsonl file