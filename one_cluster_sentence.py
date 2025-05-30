import json
import random
from load_dataset import  load_totto_dataset, load_webnlg_dataset
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



def main(n_clusters = 10,dataset_name='e2e'):

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
    



    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    print('迭代次数: ', kmeans.n_iter_)
    labels = kmeans.labels_


    # 初始化一个字典来存储每个类别的数据
    categorized_data = {}

    # 遍历数据和标签，按标签分类数据
    idx=0
    for data_left,data_right, label in zip(left, right,labels):
        if str(label) not in categorized_data.keys():
            categorized_data[str(label)] = []
        categorized_data[str(label)].append({'idx':idx,'left':data_left,'right':data_right})
        idx+=1
    # 将分类后的数据字典保存为JSON文件
    with open('cluster_result_sentence_bert/one_cluster_random/'+dataset_name+'_' + str(n_clusters)+'.json', 'w',encoding='utf-8') as f:
        json.dump(categorized_data, f, ensure_ascii=False)





                                                                                                                                                                                                                                                                                        




import numpy as np
from sklearn.cluster import KMeans




from sentence_transformers import SentenceTransformer
if __name__ == '__main__':
        

    model = SentenceTransformer('./encoder/all-MiniLM-L6-v2')
    model.to(device)


    # dataset_name='dart' 
    # dataset_name='webnlg'  
    # dataset_name='webnlg' 
    # dataset_name='totto' 
    # dataset_cluster_dict={
    #     'totto':-1,
    #     'webnlg':31,
    #     'dart':-1,
    #     'e2e':43
    # }
    # dataset_cluster_dict={
    #     'totto':11,
    #     'webnlg':10,
    #     'dart':10,
    #     'e2e':10
    # }

    dataset_name_list=['webnlg','e2e','dart','totto']

    dataset_cluster_dict={
        'totto':10,
        'webnlg':10,
        'dart':10,
        'e2e':10
    }


    

    
    for dataset_name in dataset_name_list:
        if(dataset_name=='webnlg'):
            X = np.load('emb_sentence_bert/webnlg_train.npy')
        elif(dataset_name=='e2e'):
            X = np.load('emb_sentence_bert/e2e_train.npy')
        elif(dataset_name=='dart'):
            X = np.load('emb_sentence_bert/dart_train.npy')
        elif(dataset_name=='totto'):
            X=np.load('emb_sentence_bert/totto_train.npy')

        n_clusters=dataset_cluster_dict[dataset_name]

        main(n_clusters = n_clusters,dataset_name=dataset_name)
