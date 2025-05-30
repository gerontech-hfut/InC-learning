import argparse
import os
import re
import warnings

warnings.filterwarnings("ignore")
from bert_score import BERTScorer
from nltk.tokenize import word_tokenize
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from load_dataset import load_e2e_dataset, load_totto_dataset
from load_dataset import load_webnlg_dataset
from load_dataset import load_dart_dataset
from rouge import Rouge
from transformers import RobertaTokenizer, RobertaModel
#from eval_method import table_text_eval


def my_eval(output,method,dataset_name,num=1):
    if dataset_name=='e2e':
        file_path='dataset/e2e/test'
        left,right=load_e2e_dataset(file_path, mode='one_left_muti_right')
    if dataset_name=='dart':
        file_path = 'dataset/dart/dart-v1.1.1-full-test.json'
        left,right = load_dart_dataset(file_path, mode='one_left_muti_right')
    if dataset_name=='webnlg':
        file_path = 'dataset/webnlg/testdata_with_lex'
        left,right = load_webnlg_dataset(file_path, mode='one_left_muti_right')
    if dataset_name == 'totto':
        file_path='dataset/ToTTo/ToTTo_dev_clean.csv'
        left,right = load_totto_dataset(file_path, mode='one_left_muti_right')
    if method=='bert_score':
        from bert_score import BERTScorer
        scorer_model=BERTScorer('roberta-large', device='cuda:0', rescale_with_baseline=True, lang='en')
        all_result = []
        for i in range(len(right)):

            result = []
            dev_lines=[]
            if not os.path.exists(output + '/' + str(i)):
                continue
            for j in range(num):
                output_dir = output + '/' + str(i) + '/result_'+str(j)+'.txt'

                dev_lines.append(open(output_dir, 'r', encoding='utf-8').read())
            for dev_line in dev_lines:
                result.append(scorer_model.score([dev_line], [right[i]])[2])
            all_result.append(max(result).numpy())
        print(np.average(all_result))
        return np.average(all_result)
    if method=='bleu':
        all_result = []
        for i in range(len(right)):

            result = []
            dev_lines=[]
            if not os.path.exists(output + '/' + str(i)):
                continue
            for j in range(num):
                output_dir = output + '/' + str(i) + '/result_'+str(j)+'.txt'

                dev_lines.append(open(output_dir, 'r', encoding='utf-8').read())
            for dev_line in dev_lines:
                candidates = word_tokenize(dev_line.lower())
                ref_lines = right[i]
                references = [word_tokenize(ref_line.lower()) for ref_line in ref_lines]
                result.append(sentence_bleu(references, candidates))
            all_result.append(max(result))
        print(np.average(all_result))
        return np.average(all_result)
    if method=='rouge':
        rouge = Rouge()
        all_result = []
        for i in range(len(right)):
            result = []
            dev_lines=[]
            if not os.path.exists(output + '/' + str(i)):
                continue
            for j in range(num):
                output_dir = output + '/' + str(i) + '/result_'+str(j)+'.txt'

                dev_lines.append(open(output_dir, 'r', encoding='utf-8').read())
            for dev_line in dev_lines:
                if dev_line=='\n':
                    dev_line='NULL'
                ref_lines = right[i]
                scores = []
                for ref_line in ref_lines:
                    scores.append(rouge.get_scores(dev_line, ref_line)[0]['rouge-l']['f'])
                result.append(max(scores))
            all_result.append(max(result))

        print(np.average(all_result))
        return np.average(all_result)





def batch2list(output_path,dataset_name,eval_method):
    if dataset_name=='dart':
        file_path = 'dataset/dart/dart-v1.1.1-full-test.json'
        left,right = load_dart_dataset(file_path, mode='one_left_muti_right')
    if dataset_name=='webnlg':
        file_path = 'dataset/webnlg/testdata_with_lex'
        left,right = load_webnlg_dataset(file_path, mode='one_left_muti_right')
    if dataset_name == 'e2e':
        file_path = 'dataset/e2e/test'
        left,right = load_e2e_dataset(file_path, mode='one_left_muti_right')
    if dataset_name == 'totto':
        file_path='dataset/ToTTo/ToTTo_dev_clean.csv'
        left,right = load_totto_dataset(file_path, mode='one_left_muti_right')
    if test_number<len(left):
        new_left=[]
        new_right=[]
        mod_temp = int(len(left)/test_number)
        seed=1
        count=0
        for one_input_idx in range(len(left)):
            if not test_number>len(left):
                if not one_input_idx % mod_temp==seed:
                    continue
                new_left.append(left[one_input_idx])
                new_right.append(right[one_input_idx])
                count+=1
        left=new_left
        right=new_right
    result_list=[]
    if eval_method=='rouge':
        rouge = Rouge()
    if eval_method=='bert_score':
        scorer_model=BERTScorer('roberta-large', device='cuda:0', rescale_with_baseline=True, lang='en')
    for i in range(len(right)):
        result_list.append([])
    for one_batch_path in os.listdir(output_path):
        idx_list=open(output_path+'/'+one_batch_path+'/test_idx.txt','r',encoding='utf-8').readlines()
        
        ori_text_list=open(output_path+'/'+one_batch_path+'/result_0.txt','r',encoding='utf-8').readlines()
        text_list=[]
        for ori_text in ori_text_list:
            if len(ori_text)<5:
                continue
            text_list.append(ori_text)
        if len(text_list)==len(idx_list):
            for idx,text in zip(idx_list,text_list):
                result_list[int(idx)].append(text)
        else:
            if len(text_list)==len(idx_list)+1:
                reduced_text_list=text_list[1:]
                for idx,text in zip(idx_list,reduced_text_list):
                    result_list[int(idx)].append(text)
                continue
            else:
                if len(text_list)==2*len(idx_list):
                    temp_text_list=[]
                    count=1
                    for text in text_list: 
                        if count%2==0:
                            temp_text_list.append(text)
                        count+=1
                    for idx,text in zip(idx_list,temp_text_list):
                        result_list[int(idx)].append(text)
                    continue
                if len(text_list)==2*len(idx_list)+1:
                    text_list=text_list[1:]
                    temp_text_list=[]
                    count=1
                    for text in text_list: 
                        if count%2==0:
                            temp_text_list.append(text)
                        count+=1
                    for idx,text in zip(idx_list,temp_text_list):
                        result_list[int(idx)].append(text)
                    continue

                print('bad file idx: '+str(one_batch_path))

    all_result=[]
    bad_result_count=0
    bad_result_list=[]
    for i in range(len(right)):
        if result_list[i]==[]:
            bad_result_list.append(i)
            result_list[i].append('NULL')
        result = []
        for dev_line in result_list[i]:
            ref_lines = right[i]
            scores = []
            for ref_line in ref_lines:
                if dev_line=='':
                    dev_line='NULL'
                if ref_line=='':
                    continue
                if eval_method=='rouge':
                    scores.append(rouge.get_scores(dev_line, ref_line)[0]['rouge-l']['f'])
                if eval_method=='bleu':
                    candidates = word_tokenize(dev_line.lower())
                    references = [word_tokenize(ref_line.lower()) for ref_line in ref_lines]
                    scores.append(sentence_bleu(references, candidates))
                if eval_method=='bert_score':
                    scores.append(scorer_model.score([dev_line], [ref_lines])[2].numpy())

            if scores==[]:
                scores=[0]
            result.append(max(scores))
        all_result.append(max(result))
    print(np.average(all_result))
    return np.average(all_result)



if __name__ == '__main__':

    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True, choices=['webnlg', 'e2e', 'dart', 'totto'], help='Name of the dataset')
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--test_number', type=int, default=100,required=True)
    parser.add_argument('--batch_size', type=int, default=1,required=True)


    args = parser.parse_args()
    output_path=args.output_path
    dataset_name = args.dataset_name
    batch_size=args.batch_size
    test_number=args.test_number
  
    if batch_size==1:

        bleu_result=my_eval(output=output_path,method='bleu',dataset_name=dataset_name)
        rouge_result=my_eval(output=output_path,method='rouge',dataset_name=dataset_name)
        bert_score_result=my_eval(output=output_path,method='bert_score',dataset_name=dataset_name)
    else:
        bleu_result=batch2list(output_path,dataset_name=dataset_name,eval_method='bleu')

        