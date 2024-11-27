
work_dir=preprocess
data_pool_input="your_retrieval_pool_data"  # data should with tag
data_pool_embed_input="your_retrieval_pool_embed_data" 
whole_train_tag_input="your_whole_domain_tag_data"   # data should with tag

create_output_dir() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
    fi
}

# 执行检索任务
run_retrieval() {
    output_dir=$1
    error_data_file=$2
    
    retrieval_types=("tag_similarity")
    GPUS=($3)  

    retrieval_scales=1,2,3

    create_output_dir "$output_dir"

    for ((i=0; i<${#retrieval_types[@]}; i++)); do
        export CUDA_VISIBLE_DEVICES=${GPUS[$i]}
        python $work_dir/retrieval.py \
            --error_data_input $error_data_file \
            --whole_train_tag_input $whole_train_tag_input \
            --data_pool_input $data_pool_input \
            --data_pool_embed_input $data_pool_embed_input \
            --retrieval_scales $retrieval_scales \
            --encode_type Q_type \
            --output $output_dir \
            --retrieval_type ${retrieval_types[$i]} 
    done
}


output=data/retrieval
error_data_path=your_error_data_file
run_retrieval $output $error_data_path "0" &

