import argparse
import json
from sentence_transformers import SentenceTransformer,util
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import numpy as np
import copy
import os
import random
from tqdm import tqdm
import math

def parse_args():

    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--error_data_input', type=str, required=True, help='error_data_input，formate is dpo type')
    parser.add_argument('--whole_train_tag_input', type=str, required=True, help='whole train data with tag')
    parser.add_argument('--data_pool_input', type=str, required=True, help='retrieval data pool, sft type')
    parser.add_argument('--data_pool_embed_input', type=str, help='retrieval data pool embedding')
    parser.add_argument('--output', type=str, required=True, help='output dir')
    parser.add_argument('--retrieval_scales', type=str, default="1", help='retrieval scale, usage 1,2,3')
    parser.add_argument('--retrieval_type', type=str, default="knn", help='retrieval method')
    parser.add_argument('--encode_type', type=str, default="Q_type", help='embedding endcode type, usage: Q_type or QA_type')
    parser.add_argument('--sentence_model_path', type=str, default="sentence-transformers/all-MiniLM-L6-v2", help='embedding model')
    parser.add_argument('--k_cluster', type=int, default=10, help='cluster num')
    parser.add_argument('--retrieval_domain',action="store_true", help="retrieval just in domain or not")


    return parser.parse_args()

def load_data_mode1(data_file):
    data = json.load(open(data_file, "r",encoding='utf-8'))
    return data
def save_data_mode1(data,data_file):
    with open(data_file,"w",encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
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
            raise RuntimeError("error encode type")
    if mode == "Q_type":
        print("just use question do k-mean")
    elif mode == "QA_type": # question + answer
        print("use both question and answer")

    return new_all

def embedding_convert(datas,model,encode_type):
    datas_convert = convert_data_type(datas,encode_type)
    datas_embed = model.encode(datas_convert,batch_size=32,show_progress_bar=True)
    return datas_embed

def dedup_data(all_data):

    unique_prompts = {}
    for data in all_data:
        prompt_id = data["id"]
        if prompt_id not in unique_prompts:
            unique_prompts[prompt_id] = data

    filtered_data = list(unique_prompts.values())
    return filtered_data

def add_tag_unique(whole_data,datas):
    whole_data_index = {item['id']: item for item in whole_data}
    for idx,item in enumerate(datas):
        item["unique_id"] = idx
        if item['prompt_id'] in whole_data_index:

            whole_data_item = whole_data_index[item['prompt_id']]
            if 'tag' in whole_data_item: 
                item['tag'] = whole_data_item['tag']
    return datas

def extrac_tag(error_data):

    new_tag_index = {}
    for item in error_data:
        try:
            item['tag'] = json.loads(item['tag'])
        except json.JSONDecodeError:
            item['tag'] = None
            continue

        if item['tag'] is None:
            continue

        for tag in item['tag']:
            if tag not in new_tag_index:
                new_tag_index[tag] = []
            new_tag_index[tag].append(item['unique_id'])  
    return new_tag_index

def retrieval_knn(error_data_embed,error_data,data_pool_embed,data_pool,retrieval_scales,k_cluster):

    kmeans = KMeans(n_clusters=k_cluster, random_state=42) 
    error_labels = kmeans.fit_predict(error_data_embed)

    cluster_centers = kmeans.cluster_centers_
    similarities = cosine_similarity(data_pool_embed, cluster_centers)


    error_ids = {d['prompt_id'] for d in error_data}

    all_retrieval_data = []

    selected_indices = [[] for _ in range(len(retrieval_scales))]
    for i in tqdm(range(cluster_centers.shape[0])):
        cluster_similarities = similarities[:, i]
        num_samples_in_cluster = np.sum(error_labels == i)
        sorted_indices = np.argsort(-cluster_similarities)
        
        filtered_indices = [idx for idx in sorted_indices if data_pool[idx]['id'] not in error_ids]

        for j,retrieval_scale in enumerate(retrieval_scales):
            top_indices = filtered_indices[:num_samples_in_cluster * retrieval_scale]
            selected_indices[j].extend(top_indices)


    for j in range(len(retrieval_scales)):
        selected_general_data = [data_pool[i] for i in selected_indices[j]]
        filter_data = dedup_data(selected_general_data)
        all_retrieval_data.append(filter_data)


    return all_retrieval_data

def retrieval_similarity(error_data_embed,error_data,data_pool_embed,data_pool,retrieval_scales):

    all_retrieval_data = []
    error_mean_embed = np.mean(error_data_embed, axis=0)

    error_mean_embed = np.expand_dims(error_mean_embed, axis=0)

    similarity_scores = cosine_similarity(error_mean_embed, data_pool_embed)[0]

    sorted_indices = np.argsort(similarity_scores)[::-1]

    error_ids = {d['prompt_id'] for d in error_data}

    filtered_indices = [idx for idx in sorted_indices if data_pool[idx]['id'] not in error_ids]

    
    for retrieval_scale in retrieval_scales:
        retrieval_num = retrieval_scale * len(error_data)
        selected_general_data = [data_pool[i] for i in filtered_indices[:retrieval_num]]
        all_retrieval_data.append(selected_general_data)

    print("finsh sorted data by similarity")

    return all_retrieval_data


def retrieval_K_similarity(error_data_embed, error_data, data_pool_embed, data_pool, retrieval_scales):
    all_retrieval_data = [[] for _ in range(len(retrieval_scales))]
    
    error_ids = {d['prompt_id'] for d in error_data}
    
    similarity_matrix = np.empty((len(error_data_embed), len(data_pool_embed)))
    
    batch_size = 100  
    for start_idx in tqdm(range(0, len(error_data_embed), batch_size), desc="Calculating Cosine Similarity"):
        end_idx = min(start_idx + batch_size, len(error_data_embed))
        similarity_matrix[start_idx:end_idx] = cosine_similarity(error_data_embed[start_idx:end_idx], data_pool_embed)
    
    for i, similarity_scores in enumerate(tqdm(similarity_matrix, desc="Processing Data")):
        sorted_indices = np.argsort(similarity_scores)[::-1]
        
        filtered_indices = [idx for idx in sorted_indices if data_pool[idx]['id'] not in error_ids]
        
        for j, retrieval_scale in enumerate(retrieval_scales):
            selected_general_data = [data_pool[idx] for idx in filtered_indices[:retrieval_scale]]
            all_retrieval_data[j].extend(selected_general_data)
    
    print("Finished sorting data by K-similarity")
    

    for i in range(len(all_retrieval_data)):
        all_retrieval_data[i] = dedup_data(all_retrieval_data[i])
    
    return all_retrieval_data

def retrieval_tag(whole_train_tag,error_data_embed, error_data, data_pool_embed, data_pool, retrieval_scales):

    for idx,data in enumerate(data_pool):
        data['unique_id'] = idx
    tag_index = {}
    for item in data_pool:
        for tag in item['tag']:
            if tag not in tag_index:
                tag_index[tag] = []
            tag_index[tag].append(item['unique_id'])  
    
    print("finish making tag index")

    error_data = add_tag_unique(whole_train_tag,error_data)

    error_tag_index = extrac_tag(error_data)

    relevant_data_indices = {}
    for tag in error_tag_index:
        if tag in tag_index:
            relevant_data_indices[tag] = tag_index[tag]

    tag_cosine_similarities = {}

    for tag, indices in tqdm(error_tag_index.items(),desc=""):
        if tag in relevant_data_indices:

            mean_error_embed = np.mean(error_data_embed[indices], axis=0)

            relevant_indices = relevant_data_indices[tag]
            relevant_embeds = data_pool_embed[relevant_indices]

            cosine_similarities = cosine_similarity(mean_error_embed.reshape(1,-1), relevant_embeds).flatten()

            tag_cosine_similarities[tag] = cosine_similarities,relevant_indices

    sorted_cosine_similarities = {}
    for tag,(cosine_similarities,relevant_indices) in tqdm(tag_cosine_similarities.items(),desc=""):

        sorted_indices = np.argsort(cosine_similarities)[::-1]
        sorted_cosine_similarities[tag] = [relevant_indices[i] for i in sorted_indices]


    final_selected_indices = [[] for _ in retrieval_scales]
    total_error_data_length = sum(len(indices) for indices in error_tag_index.values())

    for tag,sorted_indices in sorted_cosine_similarities.items():
        error_tag_length = len(error_tag_index[tag])
        weight = error_tag_length / total_error_data_length

        for i,retrieval_scale in enumerate(retrieval_scales):

            num_to_select = math.ceil((weight * len(error_data) * retrieval_scale))
            num_to_select = max(1,num_to_select)

            final_selected_indices[i].extend(sorted_indices[:num_to_select])
    

    final_selected_data = []
    for indices in final_selected_indices:
        unique_indices = list(set(indices))
        selected_data = [data_pool[idx] for idx in unique_indices]
        final_selected_data.append(selected_data)
    
    return final_selected_data


def main():

    args = parse_args()
    retrieval_scales = [int(x) for x in args.retrieval_scales.split(',')]


    model = SentenceTransformer(args.sentence_model_path)
    print("finished load model")


    error_data = load_data_mode2(args.error_data_input)
    data_pool = load_data_mode2(args.data_pool_input)
    whole_train_tag = load_data_mode2(args.whole_train_tag_input)
    print("finished load data")


    if args.retrieval_domain:
        origin_error_data = add_tag_unique(whole_train_tag,error_data)
        _ = extrac_tag(origin_error_data)
        print("now only retrieval domain data")
        print(f"befor fillter {len(error_data)}")

        filtered_errordata = [item for item in error_data if any(keyword in item['prompt_id'] 
                    for keyword in ['gsm8k', 'code_alpaca', 'dolly'])]
        error_data = filtered_errordata.copy()
        print(f"after fillter {len(error_data)}")
        
        

    error_data_embed = embedding_convert(error_data,model,args.encode_type)
    
        
    data_pool_embed = np.load(args.data_pool_embed_input)

    print("finished convert data")

    # 检索
    if args.retrieval_type == "knn":

        retrieval_datas = retrieval_knn(error_data_embed,error_data,
                                     data_pool_embed,data_pool,
                                     retrieval_scales,args.k_cluster)
    elif args.retrieval_type == "similarity":

        retrieval_datas = retrieval_similarity(error_data_embed,error_data,
                            data_pool_embed,data_pool,
                            retrieval_scales)

        
    elif args.retrieval_type == "K_similarity":

        retrieval_datas = retrieval_K_similarity(error_data_embed,error_data,
                            data_pool_embed,data_pool,
                            retrieval_scales)

    elif args.retrieval_type == "tag_similarity":

        retrieval_datas = retrieval_tag(whole_train_tag,error_data_embed,error_data,
                            data_pool_embed,data_pool,
                            retrieval_scales)

    else:
        retrieval_datas = None
        raise RuntimeError("use knn or similarity or K_similarity")

    random.seed(42)

    for j,retrieval_data in enumerate(retrieval_datas):
        new_retrieval_data = []
        for i,r_data in tqdm(enumerate(retrieval_data)):
            rand_num = random.randint(0,len(retrieval_data)-1)
            while i == rand_num:
                rand_num = random.randint(0,len(retrieval_data)-1)

            new_retrieval_data.append({
                "prompt_id": r_data["id"],
                "prompt": r_data["messages"][0]["content"],
                "chosen":[
                    {"role":"user","content":r_data["messages"][0]["content"],},
                    {"role":"assistant","content":r_data["messages"][1]["content"]}
                ],
                "rejected":[
                    {"role":"user","content":r_data["messages"][0]["content"],},
                    {"role":"assistant","content":retrieval_data[rand_num]["messages"][1]["content"]}
                ],
                "messages":[
                    {"role":"user","content":r_data["messages"][0]["content"],},
                    {"role":"assistant","content":r_data["messages"][1]["content"]}
                ],
                "score_chosen":5.0,
                "score_rejected":5.0,
                "tag":r_data["tag"],
            })

        if args.retrieval_domain:
            new_retrieval_data += origin_error_data
        else:
            new_retrieval_data += error_data

        for data in new_retrieval_data:
            if "unique_id" in data:
                del data["unique_id"]
            if "tag" in data:
                del data["tag"]
        
        if not os.path.exists(args.output):
            os.makedirs(args.output)

        base_name = os.path.basename(args.error_data_input)
        file_name, _ = os.path.splitext(base_name)
        retrieval_scale = retrieval_scales[j]
        save_data_mode2(new_retrieval_data,os.path.join(args.output,f"{file_name}_retrieval_{args.retrieval_type}_{retrieval_scale}.jsonl"))
        print("finished save data")

if __name__ == "__main__":
    main()
