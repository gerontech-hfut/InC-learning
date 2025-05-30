import argparse
import random
import re
import time
import httpx
from openai import OpenAI
import torch
import json
from sklearn.metrics import pairwise
import math
import faiss
from tqdm import tqdm
import numpy as np
import os
from load_dataset import load_e2e_dataset, load_totto_dataset,load_webnlg_dataset,load_dart_dataset
import joblib
from scipy.spatial.distance import cosine
import transformers
from transformers import RobertaTokenizer, RobertaModel

device='cuda:0'
def chunks(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

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

        del sentence_batch, attn_mask, embedding_output_batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    return np.concatenate(embeddings, axis=0)

def ask_llama31(prompt,output_dir,temperature=0,result_num=1):
    messages = [
        {"role": "user", "content": prompt},
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=1024,
        temperature=0.01
    )
    count=0
    text_result=outputs[count]["generated_text"][-1]['content']
    print(text_result)
    f_prompt = open(output_dir + '/prompt.txt', 'w', encoding='utf-8')
    f_prompt.write(prompt)
    f_prompt.close()
    f_output = open(output_dir + '/result_' + str(count) + '.txt', 'w', encoding='utf-8')
    f_output.write(text_result+ '\n')
    f_output.close()
    return 1

def get_input(dataset_name,valid_or_test='test'):
    if dataset_name=='e2e':
        from load_dataset import load_e2e_dataset
        if valid_or_test=='valid':
            file_path='dataset/e2e/valid'
            result=load_e2e_dataset(file_path , mode='only_left')
        else:
            file_path = 'dataset/e2e/test'
            result = load_e2e_dataset(file_path, mode='only_left')
        return result
    if dataset_name=='webnlg':
        from load_dataset import load_webnlg_dataset
        if valid_or_test=='dev':
            file_path= 'dataset/webnlg/dev'
            result=load_webnlg_dataset(file_path , mode='only_left')
        else:
            file_path = 'dataset/webnlg/testdata_with_lex'
            result = load_webnlg_dataset(file_path, mode='only_left')
        return result
    if dataset_name=='dart':
        from load_dataset import load_dart_dataset
        if valid_or_test=='dev':
            file_path= 'dataset/dart/dart-v1.1.1-full-dev.json'
            result=load_dart_dataset(file_path , mode='only_left')
        else:
            file_path = 'dataset/dart/dart-v1.1.1-full-test.json'
            result = load_dart_dataset(file_path, mode='only_left')
        return result
    if dataset_name=='totto':
        from load_dataset import load_totto_dataset
        if valid_or_test=='dev':
            file_path='dataset/ToTTo/ToTTo_dev_clean.csv'
            result  = load_totto_dataset(file_path, mode='only_left')
        else:
            file_path='dataset/ToTTo/ToTTo_dev_clean.csv'
            result = load_totto_dataset(file_path, mode='only_left')
        return result
    return None


def test_batch_cluster(root_path,dataset_name='dart',cluster_num=10,batch_size=5,ICL_sample_num=5,test_number=99999,cluster_mode='DCCS_batch',instruction=''):
    ori_all_test=get_input(dataset_name,valid_or_test='test')

    count=0
    all_test=[]
    mod_temp = int(len(ori_all_test)/test_number)
    seed=1
    for one_input_idx in range(len(ori_all_test)):
        if not test_number>len(ori_all_test):
            if not one_input_idx % mod_temp==seed:
                continue
        count+=1
        all_test.append(ori_all_test[one_input_idx])
    if cluster_mode=='DCCS_batch':
        x=encode(tok, model, all_test)
        cluster_path_name='cluster_result_roberta-large'
        file_name=cluster_path_name+'/'+'idx_'+dataset_name+'_'+str(cluster_num)+'.txt'
        all_center=open(file_name,'r',encoding='utf-8').readlines()
        X=[]
        for one_center in all_center:
            X.append(emb_train[int(one_center)])
        similarity = pairwise.cosine_similarity(x,X)
        x_class=[]
        for x_idx in range(len(similarity)):
            x_class.append(np.where(similarity[x_idx]==max(similarity[x_idx]))[0][0])
        cluster_set=[]
        for i in range(cluster_num):
            cluster_set.append([])
        for i in range(len(x_class)):
            cluster_set[x_class[i]].append(i)
    if cluster_mode in ['random','part_random','one_cluster_data','one_cluster_text']:
        cluster_num=1
        cluster_set=[np.arange(len(all_test))]
    

    if cluster_mode in ['DCCS_batch']:
        path_name = root_path+'/'+dataset_name+'_batch_' +str(batch_size)+'_demonstration_'+ str(ICL_sample_num)+'_test_'+str(test_number)+ '_'+cluster_mode+ '_'+str(cluster_num)
        if not os.path.exists(path_name):
            os.mkdir(path_name)
    elif cluster_mode in ['random','part_random','one_cluster_data','one_cluster_text']:
        path_name = root_path+'/'+dataset_name+'_batch_' +str(batch_size)+'_demonstration_'+ str(ICL_sample_num)+'_test_'+str(test_number)+ '_'+cluster_mode
        if not os.path.exists(path_name):
            os.mkdir(path_name)

    batch_idx=0


    for class_type in range(cluster_num):
        batch_input_set=[]
        one_batch=[]
        loop_flag=0
        for one_input_idx in cluster_set[class_type]:
            one_batch.append(one_input_idx)
            loop_flag+=1
            if loop_flag==batch_size:
                batch_input_set.append(one_batch)
                one_batch=[]
                loop_flag=0
        if not loop_flag==0:
            batch_input_set.append(one_batch)

      
        if cluster_mode in ['DCCS_batch']:
            class_file_name=cluster_path_name+'/twolayer_table_'+dataset_name+'_'+str(cluster_num)+'_'+str(ICL_sample_num)+'_'+str(class_type)+'.json'
            ICL_examples=json.loads(open(class_file_name,'r',encoding='utf-8').read())
            table_list=[]
            sentence_list=[]
            for one_example in ICL_examples:
                table_list.append(one_example['table'])
                sentence_list.append(one_example['sentence'])
        if cluster_mode=='one_cluster_data':
            file_name='cluster_result_roberta-large/one_cluster_by_table/'+dataset_name+'_'+str(ICL_sample_num)+'/example.json'
            ICL_examples=json.loads(open(file_name,'r',encoding='utf-8').read())
            table_list=[]
            sentence_list=[]
            for one_example in ICL_examples:
                table_list.append(one_example['table'])
                sentence_list.append(one_example['sentence'])
        if cluster_mode=='one_cluster_text':
            file_name='cluster_result_roberta-large/one_cluster_by_sentence/'+dataset_name+'_'+str(ICL_sample_num)+'/example.json'
            ICL_examples=json.loads(open(file_name,'r',encoding='utf-8').read())
            table_list=[]
            sentence_list=[]
            for one_example in ICL_examples:
                table_list.append(one_example['table'])
                sentence_list.append(one_example['sentence'])
        if cluster_mode=='random':
            select_idx=random.sample(range(0,len(dataset_left)),ICL_sample_num)
            table_list=[]
            sentence_list=[]
            for one_sample_idx in select_idx:
                left=dataset_left[one_sample_idx]
                table_list.append(left)
                select_idx2=random.sample(range(0,len(dataset_right[one_sample_idx])),1)
                right=dataset_right[one_sample_idx][select_idx2[0]]
                sentence_list.append(right)
        for one_input_batch in batch_input_set:
            if cluster_mode=='part_random':
                select_idx=random.sample(range(0,len(dataset_left)),ICL_sample_num)
                table_list=[]
                sentence_list=[]
                for one_sample_idx in select_idx:
                    left=dataset_left[one_sample_idx]
                    table_list.append(left)
                    select_idx2=random.sample(range(0,len(dataset_right[one_sample_idx])),1)
                    right=dataset_right[one_sample_idx][select_idx2[0]]
                    sentence_list.append(right)

            prompt_text=instruction
            input_id=1
            for one_table,one_sentence in zip(table_list,sentence_list):
                prompt_text=prompt_text+"Input "+str(input_id)+" :\n"+one_table+'\n'
                input_id+=1
            input_id=1
            for one_table,one_sentence in zip(table_list,sentence_list):
                prompt_text=prompt_text+"Output "+str(input_id)+" :\n"+one_sentence+'\n'
                input_id+=1
            prompt_text = prompt_text+"\nNext, Generate the text corresponding to the following "+str(len(one_input_batch))+" tables, and the format is the same as Demonstration.\n"
            input_id=1
            for one_input_table_idx in one_input_batch:
                prompt_text = prompt_text+"Input "+str(input_id)+" :\n"+all_test[one_input_table_idx]+'\n'
                input_id+=1
            output_dir=path_name+'/'+str(batch_idx)
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
    
            ask_llama31(prompt_text,output_dir)
            f_output = open(output_dir + '/test_idx.txt', 'w', encoding='utf-8')
            for one_input_table_idx in one_input_batch:
                f_output.write(str(one_input_table_idx)+'\n')
            f_output.close()
            batch_idx+=1





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True, choices=['webnlg', 'e2e', 'dart', 'totto'], help='Name of the dataset')
    parser.add_argument('--ICL_example_num', type=int, required=True, help='Number of ICL examples to select')
    parser.add_argument('--encoder_path', type=str, required=True, help='Path to the encoder model')
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--llama_model_id',default='Meta-Llama-3.1-8B-Instruct', type=str, required=True)
    parser.add_argument('--K', type=int,default=10 ,required=True)
    parser.add_argument('--batch_ICL_method', type=str, choices=['DCCS_batch','random','part_random','one_cluster_data','one_cluster_text'],required=True)
    parser.add_argument('--batch_size', type=int, default=10,required=True)
    parser.add_argument('--test_number', type=int, default=999999,required=True)

    args = parser.parse_args()
    root_path=args.output_path
    dataset_name = args.dataset_name
    ICL_example_num = args.ICL_example_num
    encoder_name = args.encoder_path
    batch_size=args.batch_size
    llama_model_id=args.llama_model_id
    cluster_num = args.K
    ICL_mode=args.batch_ICL_method
    test_number=args.test_number

    tok = RobertaTokenizer.from_pretrained(encoder_name)
    model = RobertaModel.from_pretrained(encoder_name)
    model.to(device)
    model.eval()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)


    instruction=''

    if dataset_name=='totto':
        instruction='Put the highlighted-table together to form a sentence.\n\n\nDemonstration:\n'
    if dataset_name=='webnlg':
        instruction='Put the triples together to form a sentence.\n\n\nDemonstration:\n'
    if dataset_name=='dart':
        instruction='Put the triples together to form a sentence.\n\n\nDemonstration:\n'
    if dataset_name=='e2e':
        instruction='Put the table together to form a sentence.\n\n\nDemonstration:\n'
    if dataset_name == 'e2e':
        file_path = 'dataset/e2e/train'
        dataset_left,dataset_right = load_e2e_dataset(file_path, mode='one_left_muti_right')
    if dataset_name == 'webnlg':
        file_path = 'dataset/webnlg/train'
        dataset_left,dataset_right = load_webnlg_dataset(file_path, mode='one_left_muti_right')
    if dataset_name == 'dart':
        file_path = 'dataset/dart/dart-v1.1.1-full-train.json'
        dataset_left,dataset_right = load_dart_dataset(file_path, mode='one_left_muti_right')
    if dataset_name=='totto':
        file_path='dataset/ToTTo/ToTTo_train_clean.csv'
        dataset_left, dataset_right  = load_totto_dataset(file_path, mode='one_left_muti_right')
    emb_train = np.load('emb_roberta-large/'+dataset_name+'_train.npy')
    pipeline = transformers.pipeline(
    "text-generation",
    model=llama_model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",)
    test_batch_cluster(root_path,dataset_name=dataset_name,cluster_num=cluster_num,batch_size=batch_size,ICL_sample_num=ICL_example_num,test_number=test_number,cluster_mode=ICL_mode,instruction=instruction)