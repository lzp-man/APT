export CUDA_VISIBLE_DEVICES=0

MODEL_PATH=ckpts/dpo/llama/iterate_1-merge

OUTPUT=results/llama/dolly/iterate_1

export IS_ALPACA_EVAL_2=False

# generation 
python eval/alpaca_eval/model_generate.py --model-path $MODEL_PATH --output-dir $OUTPUT

# judge (we use chatgpt as judge model)
export OPENAI_API_KEY="your_api_key"
export OPENAI_API_BASE="your_api_base"
alpaca_eval --model_outputs $OUTPUT/model_output.json --annotators_config 'chatgpt' > $OUTPUT/result.log

