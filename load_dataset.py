import csv
import os
import json
import sys


def load_e2e_dataset(file_path , mode='one_left_muti_right'):

    all_left=[]
    all_right=[]
    for one_path in range(len(os.listdir(file_path))):
        #print(str(one_path))
        left_file_name=file_path+'/'+str(one_path)+'/left.txt'
        right_file_name = file_path + '/' + str(one_path)+ '/right.txt'
        left_file=open(left_file_name,'r',encoding='utf-8')
        right_file = open(right_file_name, 'r', encoding='utf-8')
        all_left.append(left_file.read())
        all_right.append(right_file.readlines())
    result = []
    if(mode=='one_left_one_right'):
        for (left,right) in zip(all_left,all_right):
            result.append(left+' || '+right[0])
        return result
    if(mode=='muti_left_muti_right'):
        for (left,right) in zip(all_left,all_right):
            for r in right:
                result.append(left+' || '+r)
        return result
    if(mode=='only_left'):
        return all_left
    if(mode=='one_left_muti_right'):
        return all_left,all_right
    return result



def load_webnlg_dataset(file_path , mode='one_left_muti_right'):
    all_left=[]
    all_right=[]
    for one_path in range(len(os.listdir(file_path))):
        left_file_name=file_path+'/'+str(one_path)+'/left.txt'
        right_file_name = file_path + '/' + str(one_path)+ '/right.txt'
        left_file=open(left_file_name,'r',encoding='utf-8')
        right_file = open(right_file_name, 'r', encoding='utf-8')
        all_left.append(left_file.read())
        all_right.append(right_file.readlines())
    result = []
    if(mode=='one_left_one_right'):
        for (left,right) in zip(all_left,all_right):
            result.append(left+' || '+right[0])
        return result
    if(mode=='muti_left_muti_right'):
        for (left,right) in zip(all_left,all_right):
            for r in right:
                result.append(left+' || '+r)
        return result
    if(mode=='only_left'):
        return all_left
    if(mode=='one_left_muti_right'):
        return all_left,all_right
    return result



def load_dart_dataset(file_path,mode='one_left_muti_right'):

    lines_dict=json.loads(open(file_path,'r',encoding='utf-8').read())

    result=[]
    full_src_lst = []
    full_tgt_lst = []
    for example in lines_dict:

        temp_triples = ''
        for i, tripleset in enumerate(example['tripleset']):
            subj, rela, obj = tripleset

            if i > 0:
                temp_triples += '\n'
            temp_triples += '{} | {} | {}'.format(subj, rela, obj)
        full_src_lst.append(temp_triples)
        tgt_lst=[]
        for sent in example['annotations']:
            tgt_lst.append(sent['text'])

        full_tgt_lst.append(tgt_lst)
    if (mode == 'one_left_one_right'):
        for (left, right) in zip(full_src_lst, full_tgt_lst):
            result.append(left + ' || ' + right[0])
        return result
    if (mode == 'muti_left_muti_right'):
        for (left, right) in zip(full_src_lst, full_tgt_lst):
            for r in right:
                result.append(left + ' || ' + r)
        return result
    if (mode == 'only_left'):
        return full_src_lst
    if (mode == 'one_left_muti_right'):
        return full_src_lst, full_tgt_lst


def load_totto_dataset(file_path,mode='one_left_muti_right'):
    csv.field_size_limit(2000000)


    full_src_lst = []
    full_tgt_lst = []


    with open(file_path, encoding='utf-8',newline='') as csvfile:

        reader = csv.reader(csvfile)
        next(reader)  
        for row in reader:
            if len(row) >= 2:  
                full_src_lst.append(row[0])
                full_tgt_lst.append([row[1]])
                

    if (mode == 'one_left_muti_right'):
        return full_src_lst, full_tgt_lst
    if (mode == 'only_left'):
        return full_src_lst
