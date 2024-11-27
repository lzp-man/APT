import argparse
def merge_jsonl(parts, output_file):

    with open(output_file, 'w', encoding='utf-8') as outfile:

        for part in parts:

            with open(part, 'r', encoding='utf-8') as infile:

                for line in infile:
                    outfile.write(line)


parser = argparse.ArgumentParser() 
parser.add_argument("--name")
parser.add_argument("--split",default=4)
args=parser.parse_args()
name=args.name
split=int(args.split)
parts=['']*split
for i in range(len(parts)):
    parts[i]="data/tmp/"+str(i)+".jsonl" #+parts[i]
print(parts[0])

merge_jsonl(parts, name)
