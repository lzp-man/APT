from vllm import LLM, SamplingParams
from fastchat.llm_judge.common import load_questions, temperature_config
from fastchat.model import load_model, get_conversation_template
import torch
import json
import argparse
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument("--filename")
parser.add_argument("--save_path")
parser.add_argument("--model_path")
args=parser.parse_args()
filename=args.filename
save_path=args.save_path

model_id = "qwen-1p8b-tagger"
model_path = args.model_path # your tag model path

# Load model with vLLM
model = LLM(model=model_path, trust_remote_code=True,max_model_len=2048)
# Setting greedy decoding with temperature=0
sampling_params = SamplingParams(temperature=0, max_tokens=512)


torch.manual_seed(42)

# conv = get_conversation_template(model_id)

print("Loading data...")


with open(filename, 'r', encoding='utf-8') as file:

    lines = file.readlines()
    data=[]    
    buffer=100
    tmp_len=0

    # for line in lines:
        # example = json.loads(line)
    batch_size = 10
    # bs_examples=[]
 
    
    for i in tqdm(range(0, len(lines), batch_size)):
        prompts = []
        for j in range(0,batch_size):
            if len(lines)>i+j:
                example = json.loads(lines[i+j])
                question=example["messages"][0]["content"]
                conv = get_conversation_template(model_id)
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                prompts.append(prompt)
        # print(prompts)
        tmp_len=tmp_len+batch_size

        outputs = model.generate(prompts, sampling_params)

        output_batch=[]
        for output in outputs:
            output_ids = output.outputs[0].token_ids

            # be consistent with the template's stop_token_ids
            if conv.stop_token_ids:
                stop_token_ids_index = [
                    i
                    for i, id in enumerate(output_ids)
                    if id in conv.stop_token_ids
                ]
                if len(stop_token_ids_index) > 0:
                    output_ids = output_ids[: stop_token_ids_index[0]]

            output = model.get_tokenizer().decode(
                output_ids,
                spaces_between_special_tokens=False,
            )
            if conv.stop_str and output.find(conv.stop_str) > 0:
                output = output[: output.find(conv.stop_str)]
            for special_token in model.get_tokenizer().special_tokens_map.values():
                if isinstance(special_token, list):
                    for special_tok in special_token:
                        output = output.replace(special_tok, "")
                else:
                    output = output.replace(special_token, "")
            output = output.strip()
            output_batch.append(output)

            # print(output)
        for j in range(0,batch_size):
            if len(lines)>i+j:
                example = json.loads(lines[i+j])
                example["tag"]=output_batch[j]
                data.append(example)
        if tmp_len==buffer:
            tmp_len=0
            with open(save_path, "a",encoding='utf-8') as fout:
                for sample in data:
                    fout.write(json.dumps(sample,ensure_ascii=False) + "\n")
            data=[]


with open(save_path, "a",encoding='utf-8') as fout:
    for sample in data:
        fout.write(json.dumps(sample,ensure_ascii=False) + "\n")
