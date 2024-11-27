# Relative Grading: Outputs A or B
import json
import argparse
import torch
import vllm
import os
import datasets


def load_data_mode1(data_file):
    data = json.load(open(data_file, "r",encoding='utf-8'))
    return data
def load_data_mode2(data_file):
    data = []
    with open(data_file, "r",encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data
def save_data_mode2(data, data_file):
    with open(data_file, "w",encoding="utf-8") as file:
        for item in data:
            json.dump(item, file,ensure_ascii=False)
            file.write('\n')
def save_data_mode1(data,data_file):
    with open(data_file,"w",encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def generate(args,reference_data):
    # load model
    model = vllm.LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        tokenizer_mode="auto",
        tensor_parallel_size=torch.cuda.device_count(),
        max_num_batched_tokens=4096,
        gpu_memory_utilization=0.95,
        max_model_len = 2048
    )
    # setting generate args
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
            max_tokens=1024,
            stop=["\n\n\n\n"],
        )
    prompts = [data["instruction"] for data in reference_data]
    generations = model.generate(prompts, sampling_params)
    prompt_to_output = {
        g.prompt: g.outputs[0].text for g in generations
    }
    outputs = [prompt_to_output[prompt] if prompt in prompt_to_output else "" for prompt in prompts]
    return outputs


def main(args):
    
    # load data
    eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
    
    # generate model output
    model_output_data = generate(args,eval_set)
    model_outputs = []
    for output,reference in zip(model_output_data,eval_set):
        reference["output"] = output
        reference["generator"] = os.path.basename(args.model_path)
        model_outputs.append(reference)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    save_data_mode1(model_outputs,os.path.join(args.output_dir,"model_output.json"))
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="./test")
    parser.add_argument("--model-path",type=str,default=None)
    parser.add_argument("--temperature",type=float,default=1.0)
    args = parser.parse_args()
    main(args)