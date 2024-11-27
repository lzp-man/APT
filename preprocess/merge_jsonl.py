import argparse
import os

def merge_jsonl(merge_file_name,input_dir, output_path, num_splits):

    # 确保输入目录存在
    if not os.path.exists(input_dir):
        print(f"cur dir not exist: {input_dir}")
        return

    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # 打开输出文件
    with open(output_path, 'w', encoding='utf-8') as outfile:
        # 遍历每个部分文件
        for i in range(1, num_splits + 1):
            part_file = os.path.join(input_dir, f'{merge_file_name}_part{i}.jsonl')
            if os.path.exists(part_file):
                with open(part_file, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        outfile.write(line)
                print("correct merge!")
            else:
                print(f"part file not exist: {part_file}")

def main():
    parser = argparse.ArgumentParser(description="merge all json")
    parser.add_argument("--input_dir", type=str, required=True, help="json dir")
    parser.add_argument("--merge_file_name", type=str, required=True, help="merge file name")
    parser.add_argument("--output_path", type=str, required=True, help="merge file path")
    parser.add_argument("--num_splits", type=int, required=True, help="num of merge")
    args = parser.parse_args()

    merge_jsonl(args.merge_file_name,args.input_dir, args.output_path, args.num_splits)

if __name__ == "__main__":
    main()
