import json
import os
import numpy as np
import argparse
from sentence_transformers import SentenceTransformer,util
def load_data_mode2(data_file):
    data = []
    with open(data_file, "r",encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line))
    return data
def save_data_mode(data, data_file):
    with open(data_file, "w",encoding="utf-8") as file:
        for item in data:
            json.dump(item, file,ensure_ascii=False)
            file.write('\n')
def convert_data_type(datas,mode):
    new_all = []
    for data in datas:
        if mode == "Q_type":
            new_all.append(data["messages"][0]["content"])
            # print("just use question do k-mean")
        elif mode == "QA_type" : # question + answer
            new_all.append("Question: " + data["messages"][0]["content"] + "Answer: " + data["messages"][1]["content"])
            # print("use both question and answer")
        else:
            raise RuntimeError("error embeding type")
    if mode == "Q_type":
        print("just use question do k-mean")
    elif mode == "QA_type": # question + answer
        print("use both question and answer")

    return new_all

def embedding_convert(datas,model,encode_type):
    datas_convert = convert_data_type(datas,encode_type)
    datas_embed = model.encode(datas_convert,batch_size=32,show_progress_bar=True)
    return datas_embed



def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--mode",default="Q_type")
    args = parser.parse_args()

    data = load_data_mode2(args.input_file)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    data_embed = embedding_convert(data,model,args.mode)
    np.save(args.output_file,data_embed)

if __name__ == "__main__":
    main()
