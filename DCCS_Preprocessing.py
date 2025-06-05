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
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import argparse
device='cuda:0'

def chunks(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] 
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
                similarity_matrix[i, j] = 1.0
            elif i < j:
                similarity = 1 - cosine(embeddings[i], embeddings[j])
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity

    print(similarity_matrix)

def encode(tok, model, corpus,embed_type='pooler'):
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

        del sentence_batch, attn_mask, embedding_output_batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    return np.concatenate(embeddings, axis=0)


def first_cluster(n_clusters = 10,dataset_name='e2e'):

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

    path_name='cluster_result_' + encoder_name
    if not os.path.exists(path_name):
        os.mkdir(path_name)


    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    print('iter: ', kmeans.n_iter_)
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

    print("Nearest points to the cluster centers:")
    for i, point in enumerate(nearest_points):
        print(f"Cluster {i}: {point}:{left[point]+' || '+right[point][0]}")

    filename1 = 'cluster_result_'+encoder_name+'/idx_' + dataset_name+'_' + str(n_clusters) + '.txt'
    fr = open(filename1, 'w', encoding='utf-8')
    for i in nearest_points:
        fr.write(str(i) + '\n')
    fr.close()



def second_cluster(cluster_num=10,cluster_num2=10,dataset_name='dart'):


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

    path_name='cluster_result_' + encoder_name+'/idx_'+dataset_name+'_'+str(cluster_num)+'.txt'
    first_idx=[int(i) for i in open(path_name,'r',encoding='utf-8').readlines()]


    class_list = [[] for _ in range(cluster_num)]
    for x_idx in range(len(X)):
        x=X[x_idx]
        similarity =[ 1 - cosine(x, X[fi]) for fi in first_idx]
        x_class=similarity.index(max(similarity))
        class_list[x_class].append(x_idx)
    count=0
    for one_class in class_list:
        one_class_all_sentence=[]
        one_class_all_table=[]
        for i in one_class:
            for s in right[i]:
                one_class_all_sentence.append(s)
                one_class_all_table.append(left[i])
        Y = encode(tok, model, one_class_all_sentence)

        kmeans = KMeans(n_clusters=cluster_num2)
        kmeans.fit(Y)
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

        print("Nearest points to the cluster centers:")
        for i, point in enumerate(nearest_points):
            print(f"Cluster {i}: {point}:{one_class_all_table[point]} || {one_class_all_sentence[point]}")

        filename = 'cluster_result_' + encoder_name + '/twolayer_table_' + dataset_name + '_' + str(cluster_num) +'_'+str(cluster_num2)+'_'+str(count)+'.json'
        result=[]
        fr = open(filename, 'w', encoding='utf-8')  

        for i in nearest_points:
            result.append({'table':one_class_all_table[i],'sentence':one_class_all_sentence[i]})
            
        json.dump(result,fr,ensure_ascii=False)
        fr.close()
        count=count+1
    print('over')




def eval_k(n_clusters = 10):


 
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(X, labels)

    print("silhouette score:", silhouette_avg)
    return silhouette_avg




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True, choices=['webnlg', 'e2e', 'dart', 'totto'], help='Name of the dataset')
    parser.add_argument('--ICL_example_num', type=int, required=True, help='Number of ICL examples to select')
    parser.add_argument('--encoder_path', type=str, required=True, help='Path to the encoder model')
    args = parser.parse_args()
    dataset_name = args.dataset_name
    ICL_example_num = args.ICL_example_num
    encoder_path = args.encoder_path

    tok = RobertaTokenizer.from_pretrained(encoder_path)
    model = RobertaModel.from_pretrained(encoder_path)
    model.to(device)
    model.eval()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    if(dataset_name=='webnlg'):
        X = np.load('emb_roberta-large/webnlg_train.npy')
    elif(dataset_name=='e2e'):
        X = np.load('emb_roberta-large/e2e_train.npy')
    elif(dataset_name=='dart'):
        X = np.load('emb_roberta-large/dart_train.npy')
    elif(dataset_name=='totto'):
        X=np.load('emb_roberta-large/totto_train.npy')


 
    # K=1
    # max_silhouette_score=-1
    # 
    # for k in range(5,20):
    #     temp_silhouette_score=eval_k(k)
    #     if max_silhouette_score<temp_silhouette_score:
    #         max_silhouette_score=temp_silhouette_score
    #         K=k
    dataset_cluster_dict={
        'totto':11,
        'webnlg':8,
        'dart':6,
        'e2e':19
    }
    K=dataset_cluster_dict[dataset_name]
    first_cluster(n_clusters = K,dataset_name=dataset_name)
    second_cluster(cluster_num=K, cluster_num2=ICL_example_num, dataset_name=dataset_name)

