import os
import json
import argparse
from tqdm import tqdm
from prometheus_eval.vllm import VLLM
from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE

model = VLLM(model="prometheus-7b-v2.0")
judge = PrometheusEval(model=model, absolute_grade_template=ABSOLUTE_PROMPT)

parser = argparse.ArgumentParser()
parser.add_argument("--part_name", type=str, default="")
parser.add_argument("--input_dir", type=str, default="./results", help="input dir")
parser.add_argument("--output_dir", type=str, default="./results", help="output dir")
parser.add_argument("--dpo_type_data", action="store_true", help="dpo type data process")
args = parser.parse_args()

part = args.part_name

print("Loading data...")
test_data = []
ids = []
instructions = []
responses = []
reference_answers = []
rubric_data = {
    "criteria": "You need to score the output of the model in terms of factual correctness, meeting the user's needs, logical coherence, and completeness.",
    "score1_description": "Answers provide inaccurate or incorrect information and do not fulfil the purpose and need of the user to ask the question, answers are not consistent overall or there are direct logical inconsistencies in different parts of the answer, answers do not provide enough information and important aspects are omitted.",
    "score2_description": "Answers provide inaccurate or incorrect information, fulfil the purpose and need of some users to ask questions, answers are consistent overall but there are logical inconsistencies between different parts, answers do not provide enough information and important aspects are omitted.",
    "score3_description": "The response provided inaccurate information, met the purpose and needs of the question posed by some users, fulfilled the formatting requirements of the question, and the overall logical coherence of the question was good, but the response did not provide enough information and omitted important aspects.",
    "score4_description": "The information provided in the response is accurate, fulfils the purpose and need of the question posed by some users, fulfils the formatting requirements of the question, the overall logical coherence is excellent, and the response provides sufficient information.",
    "score5_description": "The information provided in the responses was accurate and based on credible facts or data, the purpose and need of the questions posed by some users and the format of the questions were fully met, the overall logical coherence was excellent, and the responses provided sufficient information and detail."
}

input_file = os.path.join(args.input_dir, f"generate_result_{part}.jsonl")
output_file = os.path.join(args.output_dir, f"score_{part}.jsonl")

with open(input_file, 'r', encoding='utf-8') as fin:
    for line in tqdm(fin):
        example = json.loads(line)
        if args.dpo_type_data:
            ids.append(example["prompt_id"])
            instructions.append(example["prompt"])
            responses.append(example["rejected"][1]["content"])
            reference_answers.append(example["chosen"][1]["content"])
        else:
            ids.append(example["id"])
            instructions.append(example["question"])
            responses.append(example["prediction"])
            reference_answers.append(example["answer"])

score_rubric = SCORE_RUBRIC_TEMPLATE.format(**rubric_data)

feedbacks, scores = judge.absolute_grade(
    instructions=instructions,
    responses=responses,
    rubric=score_rubric,
    reference_answers=reference_answers
)

for i in range(len(instructions)):
    test_data.append(
        {
            "id": ids[i],
            "question": instructions[i],
            "answer": reference_answers[i],
            "prediction": responses[i],
            "feedback": feedbacks[i],
            "score": scores[i]
        }
    )

with open(output_file, "w", encoding='utf-8') as fout:
    for prediction in test_data:
        fout.write(json.dumps(prediction) + "\n")

print(f"Results saved to {output_file}")
