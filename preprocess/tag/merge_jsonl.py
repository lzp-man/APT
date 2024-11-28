import argparse
def merge_jsonl(parts, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for part in parts:
            with open(part, 'r', encoding='utf-8') as infile:
                for line in infile:
                    outfile.write(line)


parser = argparse.ArgumentParser() 
parser.add_argument("--output-file")
parser.add_argument("--split",default=4)
parser.add_argument("--input-dir")
args=parser.parse_args()
split=int(args.split)
parts=['']*split
for i in range(len(parts)):
    # parts[i]="data/tmp/"+str(i)+".jsonl" #+parts[i]
    parts[i] = f"{args.input_dir}/{str(i)}.jsonl"

merge_jsonl(parts, args.output_file)
