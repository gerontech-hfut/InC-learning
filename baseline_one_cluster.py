import json
import random
from load_dataset import  load_totto_dataset, load_webnlg_dataset
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm
import numpy as np
import torch
import os
from sklearn.cluster import KMeans
from load_dataset import load_e2e_dataset
from scipy.spatial.distance import cosine
from sklearn.metrics import silhouette_score
from load_dataset import load_dart_dataset

device='cuda:0'

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    return [lst[i:i + n] for i in range(0, len(lst), n)]

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def get_sim_matrix(embeddings):
    emb_len=len(embeddings)
    similarity_matrix = np.zeros((emb_len, emb_len))
    for i in range(emb_len):
        for j in range(emb_len):
            if i == j:
                # 将相同样本的相似度设置为1（最大值）
                similarity_matrix[i, j] = 1.0
            elif i < j:
                # 计算不同样本间的Cosine相似度
                similarity = 1 - cosine(embeddings[i], embeddings[j])
                similarity_matrix[i, j] = similarity
                # 由于Cosine相似度是对称的，我们可以直接填充另一半矩阵
                similarity_matrix[j, i] = similarity

    print(similarity_matrix)

def encode(tok, model, corpus,embed_type='CLS'):
    embeddings = []
    for corpus_tmp in tqdm(chunks(corpus, 32)):
        encoding = tok.batch_encode_plus(corpus_tmp, padding=True, truncation=True)
        sentence_batch, attn_mask = encoding["input_ids"], encoding["attention_mask"]
        sentence_batch, attn_mask = torch.LongTensor(sentence_batch).to(device), torch.LongTensor(attn_mask).to(
            device)

        with torch.no_grad():
            embedding_output_batch = model(sentence_batch, attn_mask)
            if embed_type == 'mean':
                sentence_embeddings = mean_pooling(embedding_output_batch, attn_mask)
            elif embed_type == 'CLS':
                sentence_embeddings = embedding_output_batch[0][:, 0, :]
            elif embed_type == 'pooler':
                sentence_embeddings = embedding_output_batch[1]
        embeddings.append(sentence_embeddings.detach().cpu().numpy())

        #         embedding_output_batch = model(sentence_batch, attn_mask)
        #         embeddings.append(embedding_output_batch[0][:, 0, :].detach().cpu())
        del sentence_batch, attn_mask, embedding_output_batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    return np.concatenate(embeddings, axis=0)




from collections import Counter

def count_elements(arr):
    # Counter对象会计算每个元素出现的次数
    return Counter(arr)





import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def one_cluster_by_sentence(output_path,cluster_num=10,dataset_name='dart'):

    if dataset_name=='dart':
        file_path= 'dataset/dart/dart-v1.1.1-full-train.json'
        left,right=load_dart_dataset(file_path, mode='one_left_muti_right')
    if dataset_name=='e2e':
        file_path='dataset/e2e/train'
        left, right = load_e2e_dataset(file_path, mode='one_left_muti_right')
    if dataset_name=='webnlg':
        file_path = 'dataset/webnlg/train'
        left, right  = load_webnlg_dataset(file_path, mode='one_left_muti_right')
    if dataset_name=='totto':
        file_path='dataset/ToTTo/ToTTo_train_clean.csv'
        left, right  = load_totto_dataset(file_path, mode='one_left_muti_right')
    encoder_name='roberta-large'
    encoder_name ='models--roberta-large/snapshots/716877d372b884cad6d419d828bac6c85b3b18d9'
    tok = RobertaTokenizer.from_pretrained(encoder_name)
    model = RobertaModel.from_pretrained(encoder_name)
    model.to(device)



    one_class_all_sentence=[]
    one_class_all_table=[]
    for i in range(len(right)):
        for s in right[i]:
            one_class_all_sentence.append(s)
            one_class_all_table.append(left[i])
    Y = encode(tok, model, one_class_all_sentence)

    kmeans = KMeans(n_clusters=cluster_num)
    kmeans.fit(Y)
    print('迭代次数: ', kmeans.n_iter_)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    nearest_points = [None] * len(cluster_centers)
    nearest_distances = [np.inf] * len(cluster_centers)


    for i, point in enumerate(Y):

        label = labels[i]
        cluster_center = cluster_centers[label]

        distance = np.linalg.norm(point - cluster_center)
        if distance < nearest_distances[label]:
            nearest_distances[label] = distance
            nearest_points[label] = i


    result_path= output_path + '/' + dataset_name + '_' + str(cluster_num)
    if not os.path.exists(result_path):
        os.mkdir(result_path)


    filename = result_path+'/example.json'
    result=[]
    fr = open(filename, 'w', encoding='utf-8')  

    for i in nearest_points:
        result.append({'table':one_class_all_table[i],'sentence':one_class_all_sentence[i]})
            
    json.dump(result,fr,ensure_ascii=False)
    
    print('over')



def one_cluster_by_table(output_path,cluster_num=10,dataset_name='dart'):

    if dataset_name=='dart':
        file_path= 'dataset/dart/dart-v1.1.1-full-train.json'
        left,right=load_dart_dataset(file_path, mode='one_left_muti_right')
    if dataset_name=='e2e':
        file_path='dataset/e2e/train'
        left, right = load_e2e_dataset(file_path, mode='one_left_muti_right')
    if dataset_name=='webnlg':
        file_path = 'dataset/webnlg/train'
        left, right  = load_webnlg_dataset(file_path, mode='one_left_muti_right')
    if dataset_name=='totto':
        file_path='dataset/ToTTo/ToTTo_train_clean.csv'
        left, right  = load_totto_dataset(file_path, mode='one_left_muti_right')
    encoder_name='roberta-large'
    encoder_name ='models--roberta-large/snapshots/716877d372b884cad6d419d828bac6c85b3b18d9'
    tok = RobertaTokenizer.from_pretrained(encoder_name)
    model = RobertaModel.from_pretrained(encoder_name)
    model.to(device)



    one_class_all_table=[]
    for i in range(len(left)):
        one_class_all_table.append(left[i])
    X = encode(tok, model, one_class_all_table)

    kmeans = KMeans(n_clusters=cluster_num)
    kmeans.fit(X)
    print('迭代次数: ', kmeans.n_iter_)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    nearest_points = [None] * len(cluster_centers)
    nearest_distances = [np.inf] * len(cluster_centers)


    for i, point in enumerate(X):

        label = labels[i]
        cluster_center = cluster_centers[label]

        distance = np.linalg.norm(point - cluster_center)
        if distance < nearest_distances[label]:
            nearest_distances[label] = distance
            nearest_points[label] = i

    result_path= output_path + '/' + dataset_name + '_' + str(cluster_num)
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    filename = result_path+'/example.json'
    result=[]
    fr = open(filename, 'w', encoding='utf-8')  

    for i in nearest_points:
        result.append({'table':one_class_all_table[i],'sentence':right[i][0]})
            
    json.dump(result,fr,ensure_ascii=False)
    
    print('over')



def cluster_random(n_clusters = 10,dataset_name='e2e'):

    if dataset_name=='dart':
        file_path= 'dataset/dart/dart-v1.1.1-full-train.json'
        left,right=load_dart_dataset(file_path, mode='one_left_muti_right')
    if dataset_name=='e2e':
        file_path='dataset/e2e/train'
        left, right = load_e2e_dataset(file_path, mode='one_left_muti_right')
    if dataset_name=='webnlg':
        file_path = 'dataset/webnlg/train'
        left, right  = load_webnlg_dataset(file_path, mode='one_left_muti_right')

    if dataset_name=='totto':
        file_path='dataset/ToTTo/ToTTo_train_clean.csv'
        left, right  = load_totto_dataset(file_path, mode='one_left_muti_right')
    

    encoder_name='roberta-large'
    encoder_name ='models--roberta-large/snapshots/716877d372b884cad6d419d828bac6c85b3b18d9'
    tok = RobertaTokenizer.from_pretrained(encoder_name)
    model = RobertaModel.from_pretrained(encoder_name)
    model.to(device)



    one_class_all_table=[]
    for i in range(len(left)):
        one_class_all_table.append(left[i])
    X = encode(tok, model, one_class_all_table)


    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    print('迭代次数: ', kmeans.n_iter_)
    labels = kmeans.labels_


    categorized_data = {}

    idx=0
    for data_left,data_right, label in zip(left, right,labels):
        if str(label) not in categorized_data.keys():
            categorized_data[str(label)] = []
        categorized_data[str(label)].append({'idx':idx,'left':data_left,'right':data_right})
        idx+=1
    with open('cluster_result_roberta-large/one_cluster_random/'+dataset_name+'_' + str(n_clusters)+'.json', 'w',encoding='utf-8') as f:
        json.dump(categorized_data, f, ensure_ascii=False)



if __name__ == '__main__':
    encoder_name ='models--roberta-large'
    tok = RobertaTokenizer.from_pretrained(encoder_name)
    model = RobertaModel.from_pretrained(encoder_name)
    model.to(device)
    model.eval()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0) 
    dataset_name_list=['webnlg','e2e','dart','totto']

    dataset_cluster_dict={
        'totto':11,
        'webnlg':8,
        'dart':6,
        'e2e':19
    }
    ICL_example_num=5

    

    
    for dataset_name in dataset_name_list:
        n_clusters=dataset_cluster_dict[dataset_name]
        cluster_random(n_clusters = n_clusters,dataset_name=dataset_name)

        output_path='cluster_result_roberta-large/one_cluster_by_sentence'
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        one_cluster_by_sentence(output_path,ICL_example_num,dataset_name)

        output_path='cluster_result_roberta-large/one_cluster_by_table'
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        one_cluster_by_table(output_path,ICL_example_num,dataset_name)

