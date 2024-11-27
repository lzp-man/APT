import argparse
import json
import os
from utils import *
def select_dpo_jsonl(file_path, save_path):
    test_data = []
    try:
        datas = load_data_mode2(file_path)

        selected_data = []
        for example in datas:
            s = example["score"]
            if s is not None and example["score"] < 3:
                selected_data.append({
                    "prompt_id": example["id"],
                    "prompt": example["question"],
                    "chosen": [
                        {"role": "user", "content": example["question"]},
                        {"role": "assistant", "content": example["answer"]}
                    ],
                    "rejected": [
                        {"role": "user", "content": example["question"]},
                        {"role": "assistant", "content": example["prediction"]}
                    ],
                    "messages": [
                        {"role": "user", "content": example["question"]},
                        {"role": "assistant", "content": example["answer"]}
                    ],
                    "score_chosen": 5.0,
                    "score_rejected": float(s) if s is not None else 0
                })
        print(f"cur score less {score}，num is {len(selected_data)}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_data_mode2(selected_data, os.path.join(save_path,f"error_case.jsonl"))

    except FileNotFoundError:
        print(f"file not found: {file_path}")
        return
    except json.JSONDecodeError:
        print(f"file decode error: {file_path}")
        return
    except Exception as e:
        print(f"read file error: {e}")
        return



def main():
    parser = argparse.ArgumentParser(description="选择符合条件的 JSONL 数据，并保存到指定路径")
    parser.add_argument("--name", required=True, help="输入的 JSONL 文件路径")
    parser.add_argument("--save-dpo", required=True, help="最终输出的 JSONL 文件路径")
    args = parser.parse_args()

    select_dpo_jsonl(args.name, args.save_dpo)
    #select_all_dpo_jsonl(args.name, args.save_all_dpo)

if __name__ == "__main__":
    main()
