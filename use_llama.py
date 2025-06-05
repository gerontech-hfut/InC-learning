import os
import sys
import time
import transformers
import torch
import json
import random
import re
import time
import httpx
import joblib
from zhipuai import ZhipuAI
import httpx
from sklearn.metrics import pairwise
import math
import faiss
from tqdm import tqdm
import numpy as np
import torch
import os
from load_dataset import load_e2e_dataset, load_totto_dataset,load_webnlg_dataset,load_dart_dataset
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.spatial.distance import cosine
from transformers import RobertaTokenizer, RobertaModel
import argparse
device='cuda:0'
  
def get_input(dataset_name,valid_or_test='test'):
  if dataset_name=='e2e':
    if valid_or_test=='valid':
      file_path='dataset/e2e/valid'
      result=load_e2e_dataset(file_path , mode='only_left')
    else:
      file_path = 'dataset/e2e/test'
      result = load_e2e_dataset(file_path, mode='only_left')
    return result
  if dataset_name=='webnlg':
    if valid_or_test=='dev':
      file_path= 'dataset/webnlg/dev'
      result=load_webnlg_dataset(file_path , mode='only_left')
    else:
      file_path = 'dataset/webnlg/testdata_with_lex'
      result = load_webnlg_dataset(file_path, mode='only_left')
    return result
  if dataset_name=='dart':
    if valid_or_test=='dev':
      file_path= 'dataset/dart/dart-v1.1.1-full-dev.json'
      result=load_dart_dataset(file_path , mode='only_left')
    else:
      file_path = 'dataset/dart/dart-v1.1.1-full-test.json'
      result = load_dart_dataset(file_path, mode='only_left')
    return result
  if dataset_name=='totto':
    if valid_or_test=='dev':
      file_path='dataset/ToTTo/ToTTo_dev_clean.csv'
      result  = load_totto_dataset(file_path, mode='only_left')
    else:
      file_path='dataset/ToTTo/ToTTo_dev_clean.csv'
      result = load_totto_dataset(file_path, mode='only_left')
    return result
  return None

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

def get_kernel(embed, candidates, scale_factor, index_global):
  near_reps = np.stack([index_global.index.reconstruct(i) for i in candidates], axis=0)
  embed = embed / np.linalg.norm(embed)
  near_reps = near_reps / np.linalg.norm(near_reps, keepdims=True, axis=1)

  rel_scores = np.matmul(embed, near_reps.T)[0]
  rel_scores = (rel_scores + 1) / 2
  rel_scores -= rel_scores.max()
  rel_scores = np.exp(rel_scores / (2 * scale_factor))
  sim_matrix = np.matmul(near_reps, near_reps.T)
  sim_matrix = (sim_matrix + 1) / 2
  kernel_matrix = rel_scores[None] * sim_matrix * rel_scores[:, None]
  return near_reps, rel_scores, kernel_matrix


def fast_map_dpp(kernel_matrix, max_length):
  item_size = kernel_matrix.shape[0]
  cis = np.zeros((max_length, item_size))
  di2s = np.copy(np.diag(kernel_matrix))
  selected_items = list()
  selected_item = np.argmax(di2s)
  selected_items.append(int(selected_item))
  while len(selected_items) < max_length:
    k = len(selected_items) - 1
    ci_optimal = cis[:k, selected_item]
    di_optimal = math.sqrt(di2s[selected_item])
    elements = kernel_matrix[selected_item, :]
    eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
    cis[k, :] = eis
    di2s -= np.square(eis)
    selected_item = np.argmax(di2s)
    selected_items.append(int(selected_item))
  return selected_items



def chunks(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] 
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def decode(tok, model, corpus,embed_type='pooler'):
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


def get_two_layer_cluster(my_input,cluster_num=10,ICL_sample_num=10,dataset_name='dart'):
    path_name = 'cluster_result_roberta-large'
    
    file_name=path_name+'/'+'idx_'+dataset_name+'_'+str(cluster_num)+'.txt'
    all_center=open(file_name,'r',encoding='utf-8').readlines()
    X=[]
    for one_center in all_center:
        X.append(emb_train[int(one_center)])
  
    x=decode(tok, model, [my_input])[0]
    similarity = pairwise.cosine_similarity([x],X)
    x_class=np.where(similarity[0]==max(similarity[0]))[0][0]
    class_file_name=path_name+'/twolayer_table_'+dataset_name+'_'+str(cluster_num)+'_'+str(ICL_sample_num)+'_'+str(x_class)+'.json'
    ICL_examples=json.loads(open(class_file_name,'r',encoding='utf-8').read())
    table_list=[]
    sentence_list=[]
    for one_example in ICL_examples:
        table_list.append(one_example['table'])
        sentence_list.append(one_example['sentence'])
    return table_list,sentence_list


def get_KFN(my_input,ICL_sample_num=10):

  emb_dev=decode(tok, model, [my_input])[0]

  dist_matrix = pairwise.cosine_similarity(X=[emb_dev], Y=emb_train)

  values, indices = torch.topk(-torch.from_numpy(dist_matrix), k=ICL_sample_num, dim=-1)

  indices = indices.numpy()[0]

  return indices

def get_BM25(my_input,ICL_sample_num=10):

  tokenized_test_text = [word for word in word_tokenize(my_input.lower()) if word not in stop_words]
  scores = bm25.get_scores(tokenized_test_text)

  sorted_indices = sorted(zip(range(0,len(scores)), scores), key=lambda x: x[1], reverse=True)

  indices =[] 
  for temp in sorted_indices[0:ICL_sample_num]:
    indices.append(temp[0])



  return indices

def get_KATE(my_input,ICL_sample_num=10):


  emb_dev=decode(tok, model, [my_input])[0]

  dist_matrix = pairwise.cosine_similarity(X=[emb_dev], Y=emb_train)

  values, indices = torch.topk(torch.from_numpy(dist_matrix), k=ICL_sample_num, dim=-1)

  indices = indices.numpy()[0]

  return indices

def get_DPR(my_input,ICL_sample_num=10):

  question_inputs = question_tokenizer(my_input, return_tensors='pt').to('cuda:0')
  question_embeddings = question_encoder(**question_inputs).pooler_output.detach().cpu().numpy()
  dist_matrix = pairwise.cosine_similarity(X=question_embeddings, Y=dpr_emb_train)

  values, indices = torch.topk(torch.from_numpy(dist_matrix), k=ICL_sample_num, dim=-1)

  indices = indices.numpy()[0]

  return indices  

def get_DPP(my_input, m, k):
  embed=np.expand_dims(decode(tok, model, [my_input])[0], axis=0)
  embed_list =emb_train
  n, d = embed_list.shape
  id_list = np.arange(n)
  index_global = faiss.IndexIDMap(faiss.IndexFlatIP(d))
  id_list = np.array([res for res in id_list])
  embed_list = np.stack([res for res in embed_list])
  index_global.add_with_ids(embed_list, id_list)
  candidates = index_global.search(embed, m)[1][0].tolist()
  near_reps, rel_scores, kernel_matrix = get_kernel(embed, candidates, 0.1, index_global)
  map_results = fast_map_dpp(kernel_matrix, k)
  ctxs_candidates = []
  for ctxs_idx in map_results:
    ctxs_candidates.append(candidates[ctxs_idx])
  return ctxs_candidates
def get_one_cluster_data(my_input,ICL_sample_num,dataset_name):
    file_name='cluster_result_sentence_bert/one_cluster_by_table/'+dataset_name+'_'+str(ICL_sample_num)+'/example.json'
    ICL_examples=json.loads(open(file_name,'r',encoding='utf-8').read())
    table_list=[]
    sentence_list=[]
    for one_example in ICL_examples:
        table_list.append(one_example['table'])
        sentence_list.append(one_example['sentence'])
    return table_list,sentence_list

def get_one_cluster_text(my_input,ICL_sample_num,dataset_name):
    file_name='cluster_result_sentence_bert/one_cluster_by_sentence/'+dataset_name+'_'+str(ICL_sample_num)+'/example.json'
    ICL_examples=json.loads(open(file_name,'r',encoding='utf-8').read())
    table_list=[]
    sentence_list=[]
    for one_example in ICL_examples:
        table_list.append(one_example['table'])
        sentence_list.append(one_example['sentence'])
    return table_list,sentence_list

def get_cluster_random(my_input,cluster_num=10,ICL_sample_num=10,dataset_name='dart'):
  path_name = 'cluster_result_sentence_bert/one_cluster_random'

  file_name=path_name+'/'+'idx_'+dataset_name+'_'+str(cluster_num)+'.txt'
  all_center=open(file_name,'r',encoding='utf-8').readlines()
  X=[]
  for one_center in all_center:
      X.append(emb_train[int(one_center)])

  x=model.encode([my_input])[0]
  similarity = pairwise.cosine_similarity([x],X)
  x_class=np.where(similarity[0]==max(similarity[0]))[0][0]
  class_file_name=path_name+'/'+dataset_name+'_'+str(cluster_num)+'.json'

  all_ICL_examples=json.loads(open(class_file_name,'r',encoding='utf-8').read())[str(x_class)]
  select_idx=random.sample(range(0,len(all_ICL_examples)),ICL_sample_num)
  table_list=[]
  sentence_list=[]
  for one_idx in select_idx:
      table_list.append(all_ICL_examples[one_idx]['left'])
      sentence_list.append(all_ICL_examples[one_idx]['right'][0])
  return table_list,sentence_list

def get_full_prompt(my_input,ICL_mode='cluster',cluster_num=10,ICL_sample_num=10,dataset_name='e2e',instruction=''):


  
  prompt_text=instruction
  if ICL_mode=='random':
    select_idx=random.sample(range(0,len(dataset_left)),ICL_sample_num)
    table=[]
    sentence=[]
    for one_sample_idx in select_idx:
      left=dataset_left[one_sample_idx]
      table.append(left)
      select_idx2=random.sample(range(0,len(dataset_right[one_sample_idx])),1)
      right=dataset_right[one_sample_idx][select_idx2[0]]
      sentence.append(right)
  elif ICL_mode=='KFN':
    indices=get_KFN(my_input, ICL_sample_num=ICL_sample_num)
    table=[]
    sentence=[]
    for one_sample_idx in indices:
      left=dataset_left[one_sample_idx]
      table.append(left)
      select_idx2=random.sample(range(0,len(dataset_right[one_sample_idx])),1)
      right=dataset_right[one_sample_idx][select_idx2[0]]
      sentence.append(right)
  elif ICL_mode=='KATE':
    indices=get_KATE(my_input, ICL_sample_num=ICL_sample_num)
    table=[]
    sentence=[]
    for one_sample_idx in indices:
      left=dataset_left[one_sample_idx]
      table.append(left)
      select_idx2=random.sample(range(0,len(dataset_right[one_sample_idx])),1)
      right=dataset_right[one_sample_idx][select_idx2[0]]
      sentence.append(right)
  
  elif ICL_mode=='DPP':
    indices=get_DPP(my_input,m=ICL_sample_num*10,k=ICL_sample_num)
    table=[]
    sentence=[]
    for one_sample_idx in indices:
      left=dataset_left[one_sample_idx]
      table.append(left)
      select_idx2=random.sample(range(0,len(dataset_right[one_sample_idx])),1)
      right=dataset_right[one_sample_idx][select_idx2[0]]
      sentence.append(right)
  elif ICL_mode=='BM25':
    indices=get_BM25(my_input, ICL_sample_num=ICL_sample_num)
    table=[]
    sentence=[]
    for one_sample_idx in indices:
      left=dataset_left[one_sample_idx]
      table.append(left)
      select_idx2=random.sample(range(0,len(dataset_right[one_sample_idx])),1)
      right=dataset_right[one_sample_idx][select_idx2[0]]
      sentence.append(right)
  elif ICL_mode=='DPR':
    indices=get_DPR(my_input, ICL_sample_num=ICL_sample_num)
    table=[]
    sentence=[]
    for one_sample_idx in indices:
      left=dataset_left[one_sample_idx]
      table.append(left)
      select_idx2=random.sample(range(0,len(dataset_right[one_sample_idx])),1)
      right=dataset_right[one_sample_idx][select_idx2[0]]
      sentence.append(right)
  elif ICL_mode=='one_cluster_data':
    table,sentence=get_one_cluster_data(my_input,ICL_sample_num=ICL_sample_num,dataset_name=dataset_name)
  elif ICL_mode=='one_cluster_text':
    table,sentence=get_one_cluster_text(my_input,ICL_sample_num=ICL_sample_num,dataset_name=dataset_name)
  elif ICL_mode=='cluster_random':
    table,sentence=get_cluster_random(my_input,cluster_num=cluster_num,ICL_sample_num=ICL_sample_num,dataset_name=dataset_name)

  elif ICL_mode=='DCCS':
    table,sentence=get_two_layer_cluster(my_input,cluster_num=cluster_num,ICL_sample_num=ICL_sample_num,dataset_name=dataset_name)
  for one_table,one_sentence in zip(table,sentence):
      prompt_text=prompt_text+'\nInput:\n'+one_table+'\nOutput:\n'+one_sentence
  prompt_text = prompt_text+"\nInput:\n"+my_input+'\nOutput:\n'

  return prompt_text





import sys
def main(output_path,dataset_name,ICL_mode='random',cluster_num=10,ICL_sample_num=10,test_number=999999,instruction="Summarize the following table in one sentence ||"):
  prompt_cost_time=[]
  chatgpt_cost_time = []
  token_cost=[]
  if not os.path.exists(output_path):
    os.mkdir(output_path)
  all_input=get_input(dataset_name, valid_or_test='test')
  count=0

  mod_temp = int(len(all_input)/test_number)
  seed=1
  for one_input_idx in range(len(all_input)):
    if not test_number>len(all_input):
      if not one_input_idx % mod_temp==seed:
        continue
    count+=1
    if count>test_number:
      break
    print(one_input_idx)

    my_input=all_input[one_input_idx]
    prompt_generate_start=time.time()

    full_prompt=get_full_prompt(my_input,ICL_mode=ICL_mode,cluster_num=cluster_num,ICL_sample_num=ICL_sample_num,dataset_name=dataset_name,instruction=instruction)
    prompt_generate_end = time.time()

    prompt_cost_time.append(prompt_generate_end-prompt_generate_start)

    
    output_dir=output_path+'/'+str(one_input_idx)
    if not os.path.exists(output_dir):
      os.mkdir(output_dir)

    gpt_access_start=time.time()
    result=ask_llama31(full_prompt,output_dir)
    gpt_access_end=time.time()
    chatgpt_cost_time.append(gpt_access_end-gpt_access_start)
    token_cost.append(result)
    if result==0:
      sys.exit()
    time.sleep(3)
  print(prompt_cost_time)
  print(chatgpt_cost_time)
  print(token_cost)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset_name', type=str, required=True, choices=['webnlg', 'e2e', 'dart', 'totto'], help='Name of the dataset')
  parser.add_argument('--ICL_example_num', type=int, required=True, help='Number of ICL examples to select')
  parser.add_argument('--encoder_path', type=str, required=True, help='Path to the encoder model')
  parser.add_argument('--output_path', type=str, required=True)
  parser.add_argument('--llama_model_id',default='Meta-Llama-3.1-8B-Instruct', type=str, required=True)
  parser.add_argument('--K', type=int,default=10 ,required=True)
  parser.add_argument('--ICL_method', type=str, choices=['random','KATE','KFN','DPP','BM25','DPR','DCCS','one_cluster_data','one_cluster_text','cluster_random'],required=True)
  parser.add_argument('--test_number', type=int, default=100,required=True)

  args = parser.parse_args()
  root_path=args.output_path
  dataset_name = args.dataset_name
  ICL_example_num = args.ICL_example_num
  encoder_name = args.encoder_path
  llama_model_id=args.llama_model_id
  if not os.path.exists(root_path):
    os.mkdir(root_path)
  tok = RobertaTokenizer.from_pretrained(encoder_name)
  model = RobertaModel.from_pretrained(encoder_name)
  model.to(device)
  model.eval()
  random.seed(0)
  np.random.seed(0)
  torch.manual_seed(0)
  if torch.cuda.is_available():
      torch.cuda.manual_seed_all(0)
  cluster_num = args.K
  ICL_mode=args.ICL_method
  test_number=args.test_number
  
  if dataset_name=='totto':
    instruction='Put the highlighted-table together to form a sentence.\n'
  if dataset_name=='webnlg':
    instruction='Put the triples together to form a sentence:\n'
  if dataset_name=='dart':
    instruction='Put the triples together to form a sentence:\n'
  if dataset_name=='e2e':
    instruction='Put the table together to form a sentence.\n'
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
  if dataset_name=='totto':
    emb_train = np.load('emb_roberta-large_0516/'+dataset_name+'_train.npy')
  if ICL_mode == 'BM25':
    train_texts = dataset_left
    stop_words = set(stopwords.words('english'))

    tokenized_train_texts = [
      [word for word in word_tokenize(doc.lower()) if word not in stop_words] for doc in train_texts
    ]
    bm25 = BM25Okapi(tokenized_train_texts)
  if ICL_mode=='DPR':
    from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
    question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base').to('cuda:0')
    question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
    dpr_emb_train = np.load('emb_roberta-large/'+dataset_name+'_drt_train.npy')

  if ICL_mode=='DCCS':
    output_path=root_path+'/'+dataset_name+'_'+ICL_mode+ '_' + str(cluster_num)+'_'+str(ICL_example_num)+'_test_'+str(test_number)
  else:
    output_path = root_path+'/' + dataset_name + '_' + ICL_mode + '_' + str(ICL_example_num) + '_test_' + str(
      test_number)
  
  pipeline = transformers.pipeline(
        "text-generation",
        model=llama_model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",)

  main(output_path,dataset_name,ICL_mode=ICL_mode,cluster_num=cluster_num,ICL_sample_num=ICL_example_num,test_number=test_number,instruction=instruction)
  print(output_path)




