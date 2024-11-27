import argparse
def split_jsonl(file_path, num_splits=4):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    split_size = len(lines) // num_splits

    for i in range(num_splits):
        file_name=file_path.split(".")[0].split("/")[-1]
        file_name = "data/tmp/"+file_name+"_part"+str(i)+".jsonl"
        with open(file_name, 'w', encoding='utf-8') as split_file:

            for line in lines[i*split_size:(i+1)*split_size if i < num_splits - 1 else None]:
                split_file.write(line)

parser = argparse.ArgumentParser()
parser.add_argument("--name")
parser.add_argument("--split")
args=parser.parse_args()
name=args.name
split=int(args.split)

split_jsonl(name,num_splits=split)
