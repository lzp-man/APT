
name=your_sft_file_name # sft file_name, put in data
num_gpu=4
input_file=data/${name}.jsonl

rm -rf data/tmp/*
python preprocess/split_jsonl.py --name $input_file --split $num_gpu

for ((i=0;i<$num_gpu; i++ ));do
    j=$(($i+1))
    CUDA_VISIBLE_DEVICES=$i python preprocess/get_tag_jsonl.py --filename $output_dir/tmp/${name}_part$i.jsonl \
	    --save_dir data/tmp/$i.jsonl &
done

wait
python preprocess/merge_jsonl.py --name data/${name}_tag.jsonl
