# Setup
We provide a `requirements.txt` file for installing the required packages. You can install the required packages using the following command:
```
conda create -n your_env python==3.10
cd open-instruct-capt
pip install -r requirements.txt
```

# Data Preparation
We provide several scripts for reproducing the whole iteration of our paper method.
## Bad Case Generation and Score
Before getting started, ensure that the environment is set up and the SFT data is prepared. The SFT-type data should be formatted as follows:
```
{
    "id": "unique_id",
    "messages": [
        {"role": "system", "content": "message_text"}, # optional
        {"role": "user", "content": "message_text"},
        {"role": "assistant", "content": "message_text"},
        {"role": "user", "content": "message_text"},
        {"role": "assistant", "content": "message_text"},
        ...
    ],
}
```
```
bash myscript/process/generate_and_score.sh
```

After executing this script, you will obtain a collection of low-scoring error data. The default name is `error_case.jsonl`.


# Retrieval step
Before using the Tag-based Similarity retrieval method, you need to label the retrieval pool data and all domains to be tested using the tag model.
```
bash myscript/process/tag.sh
```

To accelerate the retrieval process, we will pre-generate embeddings for the data in the retrieval pool. You can use the following command to generate the embeddings:
```
python preprocess/embed.py --input-file retrieval_pool_data -- output-file retrieval_pool_embed
```

To launch the retrieval process, you can use the following command:
```
bash myscript/process/retrieval.sh
```

# Training

After completing the retrieval step, you have obtained the preference data required for this iteration. You can then start preference training using the following command:

```bash
bash myscript/train/capt_llama.sh
bash myscript/train/capt_mistral.sh
```

# Evaluation
To evaluate the performance of the trained models, you can use the following command:
```
# eval gsm8k
bash myscript/eval/eval_gsm8k.sh
bash myscript/eval/eval_code.sh
bash myscript/eval/eval_dolly.sh
```
Note: For the evaluation of Dolly, we used Alpaca-Eval to assess its instruction-following capability.

# Acknowledgement
This project is based on modifications of the https://github.com/allenai/open-instruct.git open-source code.Thank you to the original authors for their work!

