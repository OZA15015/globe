import pickle
import os
import numpy as np

pickle_dir = os.listdir(path = '/home/oza/pre-experiment/glove/50d_dic')


def pickle_load(path):
    with open(path, mode='rb') as f:
        data = pickle.load(f)
        return data

arr = np.empty((0, 50), dtype = 'float32')

i = 0
check_list = []
for file_name in pickle_dir:
    dic = pickle_load('/home/oza/pre-experiment/glove/50d_dic/' + file_name)
    print(file_name)
    i += 1
    for mykey in dic.keys():
        if mykey not in check_list: #複数カテゴリにまたがる単語を重複させない, 1つにする
            check_list.append(mykey)
            arr = np.append(arr, dic[mykey])
            arr = arr.reshape(-1, 50)

print(arr.shape)
print(str(i) + "ファイル")
#np.save('/home/oza/pre-experiment/glove/numpy_vector/50d_wiki.npy', arr)
