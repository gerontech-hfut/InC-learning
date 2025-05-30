import json
import random
import argparse
from tqdm import tqdm
from load_dataset import load_dart_dataset, load_e2e_dataset, load_totto_dataset, load_webnlg_dataset
from transformers import RobertaTokenizer, RobertaModel
import numpy as np
import torch
import os
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
device='cuda:0'
def chunks(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]

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

    X = encode(tok, model, left)
    if not os.path.exists('emb_roberta-large'):
      os.mkdir('emb_roberta-large')
    with open('emb_roberta-large/'+dataset_name+'_train.npy', 'wb') as f:
        np.save(f,X)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


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


#baseline DPR
def DPR_emb(dataset_name='dart'):

    file_path = 'dataset/dart/dart-v1.1.1-full-train.json'
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

    context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base').to('cuda:0')
    context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

    file_path = 'dataset/dart/dart-v1.1.1-full-train.json'
    left = load_dart_dataset(file_path, mode='only_left')

    contexts = left
    print("context_embeddings start")

    context_embeddings = []
    for i in range(0, len(contexts)):
        print(i)
        inputs = context_tokenizer([left[i]], truncation=True, return_tensors='pt').to('cuda:0')
        outputs = context_encoder(**inputs).pooler_output.cpu().detach().numpy()
        context_embeddings.append(outputs[0])

    with open('emb_roberta-large/'+dataset_name+'_drt'+'_train.npy', 'wb') as f:
        np.save(f,context_embeddings)
    print("context_embeddings over")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True, choices=['webnlg', 'e2e', 'dart', 'totto'], help='Name of the dataset')
    parser.add_argument('--encoder_path', type=str, required=True, help='Path to the encoder model')
    args = parser.parse_args()

    dataset_name = args.dataset_name
    encoder_path = args.encoder_path
    
    tok = RobertaTokenizer.from_pretrained(encoder_path)
    model = RobertaModel.from_pretrained(encoder_path)
    model.to(device)
    table_emb(dataset_name=dataset_name)
