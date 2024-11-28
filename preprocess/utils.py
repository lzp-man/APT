
import json
import numpy as np
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
