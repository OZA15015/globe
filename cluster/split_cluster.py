import pickle
import os
import numpy as np
import pandas as pd

from collections import defaultdict
from gensim.models.keyedvectors import KeyedVectors
from sklearn.cluster import KMeans


pickle_dir = os.listdir(path = '/home/oza/pre-experiment/glove/300d_dic')


def pickle_load(path):
    with open(path, mode='rb') as f:
        data = pickle.load(f)
        return data

arr = np.empty((0, 300), dtype = 'float32')

n_clusters = 20
i = 0
check_list = []
column_category_list = []
category_list = []
counter = 1

for file_name in pickle_dir:
    dic = pickle_load('/home/oza/pre-experiment/glove/300d_dic/' + file_name) 
    file_name = file_name.replace('.pickle', '') #.picke削除
    column_category_list.append(file_name)
    i += 1
    for mykey in dic.keys():
        if mykey not in check_list: #複数カテゴリにまたがる単語を重複させない, 1つにする
            check_list.append(mykey)
            arr = np.append(arr, dic[mykey])
            arr = arr.reshape(-1, 300)
            category_list.append(file_name)
    if i == n_clusters:
        print(arr.shape)
        print(str(i) + "ファイル")
        kmeans_model = KMeans(n_clusters=n_clusters, verbose=1, random_state=42, n_jobs=-1)
        kmeans_model.fit(arr)
        cluster_labels = kmeans_model.labels_
        data = np.zeros((arr.shape[0], n_clusters))
        df = pd.DataFrame(data, columns = column_category_list)
        df['cluster_id'] = cluster_labels
        df['category'] = category_list

        for i in range(len(df.index)):
            idx = df.iloc[i]['category'] #特定の行を参照
            df.at[i, idx] = 1 #特定の行に代入

        clusterinfo = pd.DataFrame()
        for i in range(n_clusters):
            clusterinfo['c' + str(i)] = df[df['cluster_id'] == i].sum()
        clusterinfo = clusterinfo.drop(['cluster_id', 'category'])
        print(clusterinfo)

        my_plot = clusterinfo.T.plot(kind='bar', stacked=True, title= "Split" + str(n_clusters) + "Clusters") #, figsize=(5, 10))
        my_plot.figure.savefig('0628_pic/cluster_test' + str(counter) + '.png') 
        my_plot.set_xticklabels(my_plot.xaxis.get_majorticklabels(), rotation=0)

        my_plot.legend(bbox_to_anchor=(0, -0.1), loc='upper left', borderaxespad=0)
        my_plot.figure.savefig('0628_pic/cluster_test' + str(counter + 1) + '.png')
        check_list = []
        column_category_list = []
        category_list = []
        arr = np.empty((0, 300), dtype = 'float32')
        counter += 2
        i = 0