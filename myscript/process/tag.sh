
name=error_case # get in generate_and_score.sh
num_gpu=4
input_file=data/${name}.jsonl
tag_model=~/instagger_qwen1_8B # your tag model path
output_dir=data

rm -rf $output_dir/tmp
mkdir $output_dir/tmp
python preprocess/tag/split_jsonl.py --input $input_file --split $num_gpu --output-dir $output_dir/tmp

for ((i=0;i<$num_gpu; i++ ));do
    j=$(($i+1))
    CUDA_VISIBLE_DEVICES=$i python preprocess/tag/get_tag_jsonl.py \
        --filename $output_dir/tmp/${name}_part$i.jsonl \
        --model_path $tag_model \
	    --save_path $output_dir/tmp/$i.jsonl &
done

wait
python preprocess/tag/merge_jsonl.py --output-file $output_dir/${name}_tag.jsonl --split $num_gpu --input-dir $output_dir/tmp

rm -rf $output_dir/tmp