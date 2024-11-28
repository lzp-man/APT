import argparse
import os
import re
import json
import random
import torch
import vllm
import evaluate
from eval.utils import (
    generate_completions,
    load_hf_lm_and_tokenizer,
    query_openai_chat_model,
    dynamic_import_function,
)



def main(args):
    random.seed(42)

    print("Loading data...")
    test_data = []
    with open(args.data_dir) as fin:
        for line in fin:
            example = json.loads(line)
            if args.dpo_type_data:
                test_data.append({
                    "prompt_id": example["prompt_id"],
                    "question": example["prompt"],
                    "chosen": example["chosen"],
                    "score_chosen":example["score_chosen"],
                    "score_rejected":example["score_rejected"]
                })
            else:
                test_data.append({
                    "id": example["id"],
                    "messages":example["messages"],
                    "question": example["messages"][0]["content"],
                    "answer": example["messages"][1]["content"]
                })
            
    
    if args.max_num_examples and len(test_data) > args.max_num_examples:
        test_data = random.sample(test_data, args.max_num_examples)
        
    prompts = []
    for example in test_data:
        prompts.append(example["question"])

    if args.model_name_or_path:
        print("Loading model and tokenizer...")
        if args.use_vllm:
            model = vllm.LLM(
                model=args.model_name_or_path,
                tokenizer=args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path,
                tokenizer_mode="slow" if args.use_slow_tokenizer else "auto",
                tensor_parallel_size=torch.cuda.device_count(),
                max_num_batched_tokens=args.max_num_batched_tokens if args.max_num_batched_tokens else 4096,
                gpu_memory_utilization=0.95,
                max_model_len = 2048
            )
            if args.temperature != 0:
                sampling_params = vllm.SamplingParams(
                    temperature=args.temperature,
                    top_p=0.95,
                    max_tokens=512,
                    stop=["\n\n\n\n"],
                )
            else:
                sampling_params = vllm.SamplingParams(
                    temperature=0,
                    max_tokens=512,
                    stop=["\n\n\n\n"],
                )
            # We need to remap the outputs to the prompts because vllm might not return outputs for some prompts (e.g., if the prompt is too long)
            generations = model.generate(prompts, sampling_params)
            prompt_to_output = {
                g.prompt: g.outputs[0].text for g in generations
            }
            outputs = [prompt_to_output[prompt] if prompt in prompt_to_output else "" for prompt in prompts]
        else:
            model, tokenizer = load_hf_lm_and_tokenizer(
                model_name_or_path=args.model_name_or_path, 
                tokenizer_name_or_path=args.tokenizer_name_or_path, 
                load_in_8bit=args.load_in_8bit, 
                device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
                gptq_model=args.gptq,
                use_fast_tokenizer=not args.use_slow_tokenizer,
            )
            new_line_token = tokenizer.encode("\n\n", add_special_tokens=False)[-1] # get the last token because the tokenizer may add space tokens at the start.
            outputs = generate_completions(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=512,
                batch_size=args.eval_batch_size,
                stop_id_sequences=[[new_line_token]],
                do_sample=False,
            )
    else:
        pass

    predictions = []
    for output in outputs:
        predictions.append(output)
        

    if args.dpo_type_data:
        predictions = [{
            "prompt_id": example["prompt_id"],
            "prompt": example["question"],
            "chosen": example["chosen"],
            "rejected":[
                {"role": "user", "content": example["question"]},
                {"role": "assistant", "content": output}
            ],
            "score_chosen":example["score_chosen"],
            "score_rejected":example["score_rejected"]
        } for example, output in zip(test_data, outputs)]
    else:
        predictions = [{
            "id": example["id"],
            "messages":example["messages"],
            "question": example["question"],
            "answer": example["answer"],
            "model_output": output,
            "prediction": pred
        } for example, output, pred in zip(test_data, outputs, predictions)]

    with open(args.save_path, "w") as fout:
        for prediction in predictions:
            fout.write(json.dumps(prediction) + "\n") 
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="data/processed/"
    )
    parser.add_argument(
        "--max_num_batched_tokens", 
        type=int, 
        default=None, 
        help="maximum number of examples to evaluate."
    )
    parser.add_argument(
        "--max_num_examples", 
        type=int, 
        default=None, 
        help="maximum number of examples to evaluate."
    )
    parser.add_argument(
        "--save_path", 
        type=str, 
        default=None
    )
    parser.add_argument(
        "--model_name_or_path", 
        type=str, 
        default="/data/rj/open-instruct/ouput/7b_alpacagsminit_selectortrain_sft_bs32", 
        help="if specified, we will load the model to generate the predictions."
    )
    parser.add_argument(
        "--tokenizer_name_or_path", 
        type=str, 
        default="/data/rj/open-instruct/data/processed/train_selector", 
        help="if specified, we will load the tokenizer from here."
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If given, we will use the slow tokenizer."
    )
    parser.add_argument(
        "--openai_engine", 
        type=str, 
        default=None, help="if specified, we will use the OpenAI API to generate the predictions."
    )
    parser.add_argument(
        "--n_shot", 
        type=int, 
        default=0, 
        help="max number of examples to use for demonstration."
    )
    parser.add_argument(
        "--no_cot", 
        action="store_true", 
        help="If given, we're evaluating a model without chain-of-thought."
    )
    parser.add_argument(
        "--eval_batch_size", 
        type=int, 
        default=10, 
        help="batch size for evaluation."
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0, 
        help="temperature for generate."
    )
    parser.add_argument(
        "--load_in_8bit", 
        action="store_true", 
        help="load model in 8bit mode, which will reduce memory and speed up inference."
    )
    parser.add_argument(
        "--gptq", 
        action="store_true", 
        help="If given, we're evaluating a 4-bit quantized GPTQ model."
    )
    parser.add_argument(
        "--dpo_type_data", 
        action="store_true", 
        help="process_dpo_type_data"
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true", 
        help="If given, we will use the vllm library, which will likely increase the inference throughput."
    )
    parser.add_argument(
        "--use_chat_format", 
        action="store_true", 
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--chat_formatting_function", 
        type=str, 
        default="eval.templates.create_prompt_with_tulu_chat_format", 
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )
    args = parser.parse_args()

    # model_name_or_path and openai_engine cannot be both None or both not None.
    assert (args.model_name_or_path is None) != (args.openai_engine is None), "Either model_name_or_path or openai_engine should be specified."
    main(args)
