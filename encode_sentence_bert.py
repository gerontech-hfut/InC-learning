import json
import random

from tqdm import tqdm
from load_dataset import load_dart_dataset, load_e2e_dataset, load_totto_dataset, load_webnlg_dataset
from transformers import RobertaTokenizer, RobertaModel
import numpy as np
import torch
import os
device='cuda:0'

def table_emb(dataset_name='dart'):
    
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
    
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('./encoder/all-MiniLM-L6-v2')
    model.to(device)
    X = model.encode(left)
    # print(left[0])
    # print(right[0])
    # test_temp='hello world'
    # X=model.encode(left)
    # print(X)
    with open('emb_sentence_bert/'+dataset_name+'_train.npy', 'wb') as f:
        np.save(f,X)




table_emb(dataset_name='dart')
table_emb(dataset_name='e2e')
table_emb(dataset_name='webnlg')
table_emb(dataset_name='totto')