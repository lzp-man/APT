import argparse
import json
import os
from utils import *
def select_dpo_jsonl(file_path, save_path):
    datas = load_data_mode2(file_path)

    selected_data = []
    for example in datas:
        s = example["score"]
        if s is not None and s < 3:
            selected_data.append(example)
    save_data_mode2(selected_data, os.path.join(save_path,f"error_case.jsonl"))



def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--file_path", required=True, help="input file path")
    parser.add_argument("--save-dpo", required=True, help="output file dir")
    args = parser.parse_args()

    select_dpo_jsonl(args.file_path, args.save_dpo)

if __name__ == "__main__":
    main()
