import pickle
import os
import numpy as np

pickle_dir = os.listdir(path = '/home/oza/pre-experiment/glove/200d_dic')

def read_glove():
    embeddings_dict = {}
    with open("glove.6B/glove.6B.200d.txt", 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    return embeddings_dict


def pickle_load(path):
    with open(path, mode='rb') as f:
        data = pickle.load(f)
        return data

arr = np.empty((0, 200), dtype = 'float32')
#embeddings_dict = read_glove()
#i = 0

for file_name in pickle_dir:
    dic = pickle_load('/home/oza/pre-experiment/glove/200d_dic/' + file_name)
    print(file_name)
    for mykey in dic.keys():
        arr = np.append(arr, dic[mykey])
        arr = arr.reshape(-1, 200)
    #print(np.all(arr[0] == embeddings_dict['ammo']))
    #複数のカテゴリに属する単語があるけど, 除けていない

print(arr.shape)
    
