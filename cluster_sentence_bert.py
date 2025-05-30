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

    # 打印距离聚类中心最近的样本点
    print("Nearest points to the cluster centers:")
    for i, point in enumerate(nearest_points):
        print(f"Cluster {i}: {point}:{left[point]+' || '+right[point][0]}")

    filename1 = 'cluster_result_sentence_bert'+'/idx_' + dataset_name+'_' + str(n_clusters) + '.txt'
    fr = open(filename1, 'w', encoding='utf-8')
    for i in nearest_points:
        fr.write(str(i) + '\n')
    fr.close()



def second(cluster_num=10,cluster_num2=10,dataset_name='dart'):


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

    path_name='cluster_result_sentence_bert/idx_'+dataset_name+'_'+str(cluster_num)+'.txt'
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
        Y = model.encode( one_class_all_sentence)

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

        # 打印距离聚类中心最近的样本点
        print("Nearest points to the cluster centers:")
        for i, point in enumerate(nearest_points):
            print(f"Cluster {i}: {point}:{one_class_all_table[point]} || {one_class_all_sentence[point]}")

        filename = 'cluster_result_sentence_bert/twolayer_table_' + dataset_name + '_' + str(cluster_num) +'_'+str(cluster_num2)+'_'+str(count)+'.json'
        result=[]
        fr = open(filename, 'w', encoding='utf-8')  

        for i in nearest_points:
            result.append({'table':one_class_all_table[i],'sentence':one_class_all_sentence[i]})
            
        json.dump(result,fr,ensure_ascii=False)
        fr.close()
        count=count+1
    print('over')


from collections import Counter

def count_elements(arr):
    # Counter对象会计算每个元素出现的次数
    return Counter(arr)



def eval_k(n_clusters = 10):


    #print(X)
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    labels = kmeans.labels_
    # element_count = count_elements(labels)

    # 打印每个元素及其对应的次数
    # for element, count in element_count.items():
    #     print(f"元素 {element} 出现了 {count} 次")

        # 计算轮廓系数
    silhouette_avg = silhouette_score(X, labels)

    print("聚类结果的轮廓系数:", silhouette_avg)
    return silhouette_avg



import numpy as np
from sklearn.cluster import KMeans




from sentence_transformers import SentenceTransformer
if __name__ == '__main__':
        

    model = SentenceTransformer('./encoder/all-MiniLM-L6-v2')
    model.to(device)

    dataset_cluster_dict={
        'totto':-1,
        'webnlg':31,
        'dart':-1,
        'e2e':43
    }
    dataset_cluster_dict={
        'totto':10,
        'webnlg':10,
        'dart':10,
        'e2e':10
    }

    dataset_name_list=['webnlg','e2e','dart','totto']
    for dataset_name in dataset_name_list:
        if(dataset_name=='webnlg'):
            X = np.load('emb_sentence_bert/webnlg_train.npy')
        elif(dataset_name=='e2e'):
            X = np.load('emb_sentence_bert/e2e_train.npy')
        elif(dataset_name=='dart'):
            X = np.load('emb_sentence_bert/dart_train.npy')
        elif(dataset_name=='totto'):
            X=np.load('emb_sentence_bert/totto_train.npy')

        # for k in range(80,120):
        #     eval_k(k)
        n_clusters=dataset_cluster_dict[dataset_name]
        main(n_clusters = n_clusters,dataset_name=dataset_name)
        for cn in [5,10]:
            second(cluster_num=n_clusters, cluster_num2=cn, dataset_name=dataset_name)

# dataset_name='totto' 
# for k in range(5,20):
#         eval_k(k)
# 聚类结果的轮廓系数: 0.04132397
# 聚类结果的轮廓系数: 0.048395116
# 聚类结果的轮廓系数: 0.028210295
# 聚类结果的轮廓系数: 0.032943007
# 聚类结果的轮廓系数: 0.0303069
# 聚类结果的轮廓系数: 0.032758344
# 聚类结果的轮廓系数: 0.035889275
# 聚类结果的轮廓系数: 0.041081708
# 聚类结果的轮廓系数: 0.03863616
# 聚类结果的轮廓系数: 0.03951544
# 聚类结果的轮廓系数: 0.035378247
# 聚类结果的轮廓系数: 0.041784972
# 聚类结果的轮廓系数: 0.04184796
# 聚类结果的轮廓系数: 0.036471967
# 聚类结果的轮廓系数: 0.033365518
# 聚类结果的轮廓系数: 0.033978056
# 聚类结果的轮廓系数: 0.036631063
# 聚类结果的轮廓系数: 0.041898407
# 聚类结果的轮廓系数: 0.03699568
# 聚类结果的轮廓系数: 0.035936404
# 聚类结果的轮廓系数: 0.03927962
# 聚类结果的轮廓系数: 0.03781257
# 聚类结果的轮廓系数: 0.043593988
# 聚类结果的轮廓系数: 0.042365953
# 聚类结果的轮廓系数: 0.042695466
# 聚类结果的轮廓系数: 0.04067579
# 聚类结果的轮廓系数: 0.03597486
# 聚类结果的轮廓系数: 0.03792683
# 聚类结果的轮廓系数: 0.040706817
# 聚类结果的轮廓系数: 0.037143413
# 聚类结果的轮廓系数: 0.03926943
# 聚类结果的轮廓系数: 0.039478414
# 聚类结果的轮廓系数: 0.038766805
# 聚类结果的轮廓系数: 0.038104706
# 聚类结果的轮廓系数: 0.040849198
# 聚类结果的轮廓系数: 0.03802203
# 聚类结果的轮廓系数: 0.041530475
# 聚类结果的轮廓系数: 0.045328528
# 聚类结果的轮廓系数: 0.040249046
# 聚类结果的轮廓系数: 0.035671804
# 聚类结果的轮廓系数: 0.04441423
# 聚类结果的轮廓系数: 0.04078212
# 聚类结果的轮廓系数: 0.04069853
# 聚类结果的轮廓系数: 0.040891405
# 聚类结果的轮廓系数: 0.042672332
# 聚类结果的轮廓系数: 0.034826186
# 聚类结果的轮廓系数: 0.04208849
# 聚类结果的轮廓系数: 0.040820148
# 聚类结果的轮廓系数: 0.039185937
# 聚类结果的轮廓系数: 0.04149851
# 聚类结果的轮廓系数: 0.03845041
# 聚类结果的轮廓系数: 0.038698703
# 聚类结果的轮廓系数: 0.038918693
# 聚类结果的轮廓系数: 0.03684716
# 聚类结果的轮廓系数: 0.034910526
# 聚类结果的轮廓系数: 0.040957022
# 聚类结果的轮廓系数: 0.035667706
# 聚类结果的轮廓系数: 0.038538046
# 聚类结果的轮廓系数: 0.04165142
# 聚类结果的轮廓系数: 0.03631138
# 聚类结果的轮廓系数: 0.03832762
# 聚类结果的轮廓系数: 0.038210034
# 聚类结果的轮廓系数: 0.04275348
# 聚类结果的轮廓系数: 0.037831366
# 聚类结果的轮廓系数: 0.038382176
# 聚类结果的轮廓系数: 0.033628605
# 聚类结果的轮廓系数: 0.041779924
# 聚类结果的轮廓系数: 0.038104415
# 聚类结果的轮廓系数: 0.03606629
# 聚类结果的轮廓系数: 0.037853643
# 聚类结果的轮廓系数: 0.03549277
# 聚类结果的轮廓系数: 0.033230644
# 聚类结果的轮廓系数: 0.034359463
# 聚类结果的轮廓系数: 0.035658423
# 聚类结果的轮廓系数: 0.035388663
# 聚类结果的轮廓系数: 0.044389542
# 聚类结果的轮廓系数: 0.03298395
# 聚类结果的轮廓系数: 0.037599232
# 聚类结果的轮廓系数: 0.038745884
# 聚类结果的轮廓系数: 0.036410555
# 聚类结果的轮廓系数: 0.034443583
# 聚类结果的轮廓系数: 0.036137458
# 聚类结果的轮廓系数: 0.034503702
# 聚类结果的轮廓系数: 0.036868226
# 聚类结果的轮廓系数: 0.04193778
# 聚类结果的轮廓系数: 0.03735094
# 聚类结果的轮廓系数: 0.035318833
# 聚类结果的轮廓系数: 0.03395124
# 聚类结果的轮廓系数: 0.03744328
# 聚类结果的轮廓系数: 0.03538196
# 聚类结果的轮廓系数: 0.028418941
# 聚类结果的轮廓系数: 0.033063672
# 聚类结果的轮廓系数: 0.038402542
# 聚类结果的轮廓系数: 0.03918017
# 聚类结果的轮廓系数: 0.03628204

# dataset_name='e2e' 
# 聚类结果的轮廓系数: 0.120768264
# 聚类结果的轮廓系数: 0.13492586
# 聚类结果的轮廓系数: 0.15003815
# 聚类结果的轮廓系数: 0.17139947
# 聚类结果的轮廓系数: 0.19369146
# 聚类结果的轮廓系数: 0.20740597
# 聚类结果的轮廓系数: 0.20821138
# 聚类结果的轮廓系数: 0.22071889
# 聚类结果的轮廓系数: 0.2511515
# 聚类结果的轮廓系数: 0.26678133
# 聚类结果的轮廓系数: 0.26291674
# 聚类结果的轮廓系数: 0.29610133
# 聚类结果的轮廓系数: 0.29042187
# 聚类结果的轮廓系数: 0.30344728
# 聚类结果的轮廓系数: 0.30747333
# 聚类结果的轮廓系数: 0.33137962
# 聚类结果的轮廓系数: 0.34488457
# 聚类结果的轮廓系数: 0.32983735
# 聚类结果的轮廓系数: 0.35778135
# 聚类结果的轮廓系数: 0.36838952
# 聚类结果的轮廓系数: 0.35947827
# 聚类结果的轮廓系数: 0.37573504
# 聚类结果的轮廓系数: 0.39691308
# 聚类结果的轮廓系数: 0.36232185
# 聚类结果的轮廓系数: 0.37618494
# 聚类结果的轮廓系数: 0.41623038
# 聚类结果的轮廓系数: 0.40244132
# 聚类结果的轮廓系数: 0.41146418
# 聚类结果的轮廓系数: 0.38234755
# 聚类结果的轮廓系数: 0.4399167
# 聚类结果的轮廓系数: 0.41902524
# 聚类结果的轮廓系数: 0.4297365
# 聚类结果的轮廓系数: 0.4316593
# 聚类结果的轮廓系数: 0.42297882
# 聚类结果的轮廓系数: 0.44715095
# 聚类结果的轮廓系数: 0.3954218
# 聚类结果的轮廓系数: 0.4338155
# 聚类结果的轮廓系数: 0.4369346
# 聚类结果的轮廓系数: 0.45281035
# 聚类结果的轮廓系数: 0.43735418
# 聚类结果的轮廓系数: 0.4247718
# 聚类结果的轮廓系数: 0.4201123
# 聚类结果的轮廓系数: 0.41918078
# 聚类结果的轮廓系数: 0.43090108
# 聚类结果的轮廓系数: 0.4149765
# 聚类结果的轮廓系数: 0.43010935
# 聚类结果的轮廓系数: 0.43326405
# 聚类结果的轮廓系数: 0.40134898
# 聚类结果的轮廓系数: 0.41938877
# 聚类结果的轮廓系数: 0.4303561
# 聚类结果的轮廓系数: 0.41340274
# 聚类结果的轮廓系数: 0.42615345
# 聚类结果的轮廓系数: 0.42609638
# 聚类结果的轮廓系数: 0.42852962
# 聚类结果的轮廓系数: 0.43077543
# 聚类结果的轮廓系数: 0.4091052
# 聚类结果的轮廓系数: 0.4311557
# 聚类结果的轮廓系数: 0.4123755
# 聚类结果的轮廓系数: 0.41701102
# 聚类结果的轮廓系数: 0.41817838
# 聚类结果的轮廓系数: 0.41394886
# 聚类结果的轮廓系数: 0.41808626
# 聚类结果的轮廓系数: 0.43443072
# 聚类结果的轮廓系数: 0.41102389
# 聚类结果的轮廓系数: 0.40354785
# 聚类结果的轮廓系数: 0.4164498
# 聚类结果的轮廓系数: 0.40438947
# 聚类结果的轮廓系数: 0.3960091
# 聚类结果的轮廓系数: 0.40175575
# 聚类结果的轮廓系数: 0.406251
# 聚类结果的轮廓系数: 0.40372866
# 聚类结果的轮廓系数: 0.41290286
# 聚类结果的轮廓系数: 0.41348398
# 聚类结果的轮廓系数: 0.4025032
# 聚类结果的轮廓系数: 0.4026535
# 聚类结果的轮廓系数: 0.38847253
# 聚类结果的轮廓系数: 0.38766322
# 聚类结果的轮廓系数: 0.39926177
# 聚类结果的轮廓系数: 0.4018758
# 聚类结果的轮廓系数: 0.4213382
# 聚类结果的轮廓系数: 0.39432898
# 聚类结果的轮廓系数: 0.38152498
# 聚类结果的轮廓系数: 0.4000277
# 聚类结果的轮廓系数: 0.39811108
# 聚类结果的轮廓系数: 0.38199064
# 聚类结果的轮廓系数: 0.40571907
# 聚类结果的轮廓系数: 0.40102687
# 聚类结果的轮廓系数: 0.3879743
# 聚类结果的轮廓系数: 0.39203086
# 聚类结果的轮廓系数: 0.3934061
# 聚类结果的轮廓系数: 0.40289375
# 聚类结果的轮廓系数: 0.37676948
# 聚类结果的轮廓系数: 0.38545164
# 聚类结果的轮廓系数: 0.3831058
# 聚类结果的轮廓系数: 0.383401

#webnlg
# 聚类结果的轮廓系数: 0.05967481
# 聚类结果的轮廓系数: 0.06405431
# 聚类结果的轮廓系数: 0.07314114
# 聚类结果的轮廓系数: 0.072394855
# 聚类结果的轮廓系数: 0.079386815
# 聚类结果的轮廓系数: 0.08112151
# 聚类结果的轮廓系数: 0.07308728
# 聚类结果的轮廓系数: 0.082112305
# 聚类结果的轮廓系数: 0.08576312
# 聚类结果的轮廓系数: 0.07914001
# 聚类结果的轮廓系数: 0.09490434
# 聚类结果的轮廓系数: 0.08362356
# 聚类结果的轮廓系数: 0.09567212
# 聚类结果的轮廓系数: 0.09248614
# 聚类结果的轮廓系数: 0.08748795
# 聚类结果的轮廓系数: 0.097132735
# 聚类结果的轮廓系数: 0.092470974
# 聚类结果的轮廓系数: 0.08531828
# 聚类结果的轮廓系数: 0.096664
# 聚类结果的轮廓系数: 0.10005535
# 聚类结果的轮廓系数: 0.0933891
# 聚类结果的轮廓系数: 0.099188425
# 聚类结果的轮廓系数: 0.113473095
# 聚类结果的轮廓系数: 0.10909686
# 聚类结果的轮廓系数: 0.10905095
# 聚类结果的轮廓系数: 0.11503484
# 聚类结果的轮廓系数: 0.116913766
# 聚类结果的轮廓系数: 0.113450035
# 聚类结果的轮廓系数: 0.11744155
# 聚类结果的轮廓系数: 0.12482031
# 聚类结果的轮廓系数: 0.1274029
# 聚类结果的轮廓系数: 0.13242468
# 聚类结果的轮廓系数: 0.12766431
# 聚类结果的轮廓系数: 0.12964152
# 聚类结果的轮廓系数: 0.13506939
# 聚类结果的轮廓系数: 0.13353747
# 聚类结果的轮廓系数: 0.13695814
# 聚类结果的轮廓系数: 0.14564891
# 聚类结果的轮廓系数: 0.1420868
# 聚类结果的轮廓系数: 0.14140585
# 聚类结果的轮廓系数: 0.15146291
# 聚类结果的轮廓系数: 0.15071905
# 聚类结果的轮廓系数: 0.14908296
# 聚类结果的轮廓系数: 0.15199538
# 聚类结果的轮廓系数: 0.15621908
# 聚类结果的轮廓系数: 0.16038741
# 聚类结果的轮廓系数: 0.16306932
# 聚类结果的轮廓系数: 0.16312991
# 聚类结果的轮廓系数: 0.1600062
# 聚类结果的轮廓系数: 0.16012008
# 聚类结果的轮廓系数: 0.1650862
# 聚类结果的轮廓系数: 0.17005382
# 聚类结果的轮廓系数: 0.16884421
# 聚类结果的轮廓系数: 0.17225105
# 聚类结果的轮廓系数: 0.17120895
# 聚类结果的轮廓系数: 0.17155354
# 聚类结果的轮廓系数: 0.18025996
# 聚类结果的轮廓系数: 0.18250388
# 聚类结果的轮廓系数: 0.18243743
# 聚类结果的轮廓系数: 0.18216507
# 聚类结果的轮廓系数: 0.18223213
# 聚类结果的轮廓系数: 0.17999631
# 聚类结果的轮廓系数: 0.18802047
# 聚类结果的轮廓系数: 0.19088645
# 聚类结果的轮廓系数: 0.18938625
# 聚类结果的轮廓系数: 0.18976988
# 聚类结果的轮廓系数: 0.19746566
# 聚类结果的轮廓系数: 0.19052136
# 聚类结果的轮廓系数: 0.20512882
# 聚类结果的轮廓系数: 0.20166299
# 聚类结果的轮廓系数: 0.19837843
# 聚类结果的轮廓系数: 0.19901322
# 聚类结果的轮廓系数: 0.20265733
# 聚类结果的轮廓系数: 0.20606665
# 聚类结果的轮廓系数: 0.21125552
# 聚类结果的轮廓系数: 0.21281227
# 聚类结果的轮廓系数: 0.2161568
# 聚类结果的轮廓系数: 0.21220341
# 聚类结果的轮廓系数: 0.22057997
# 聚类结果的轮廓系数: 0.21274765
# 聚类结果的轮廓系数: 0.22095448
# 聚类结果的轮廓系数: 0.22134331
# 聚类结果的轮廓系数: 0.22031479
# 聚类结果的轮廓系数: 0.21671817
# 聚类结果的轮廓系数: 0.22187006
# 聚类结果的轮廓系数: 0.22550489
# 聚类结果的轮廓系数: 0.22710977
# 聚类结果的轮廓系数: 0.22076851
# 聚类结果的轮廓系数: 0.23140973
# 聚类结果的轮廓系数: 0.21801096
# 聚类结果的轮廓系数: 0.23028867
# 聚类结果的轮廓系数: 0.22817212
# 聚类结果的轮廓系数: 0.23193638
# 聚类结果的轮廓系数: 0.21949428
# 聚类结果的轮廓系数: 0.23295502
# 聚类结果的轮廓系数: 0.2298752
# 聚类结果的轮廓系数: 0.22726671
# 聚类结果的轮廓系数: 0.2290134
# 聚类结果的轮廓系数: 0.23174106
# 聚类结果的轮廓系数: 0.23606557
# 聚类结果的轮廓系数: 0.2418727
# 聚类结果的轮廓系数: 0.23842224
# 聚类结果的轮廓系数: 0.23547822
# 聚类结果的轮廓系数: 0.23692866
# 聚类结果的轮廓系数: 0.2482267
# 聚类结果的轮廓系数: 0.23672491
# 聚类结果的轮廓系数: 0.236508
# 聚类结果的轮廓系数: 0.23848413
# 聚类结果的轮廓系数: 0.24193694
# 聚类结果的轮廓系数: 0.24287368
# 聚类结果的轮廓系数: 0.2417268
# 聚类结果的轮廓系数: 0.24167298
# 聚类结果的轮廓系数: 0.24538183
# 聚类结果的轮廓系数: 0.2501415
# 聚类结果的轮廓系数: 0.25057158