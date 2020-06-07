import os
import numpy as np
import re
import pickle

category_dir = os.listdir(path = '/home/oza/pre-experiment/glove/Categories')
embeddings_dict = {}

with open("glove.6B/glove.6B.200d.txt", 'r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector

def pickle_dump(obj, path): #dictionaryファイル保存
    with open(path, mode='wb') as f:
        pickle.dump(obj,f)


for file_name in category_dir:
    with open("/home/oza/pre-experiment/glove/Categories/" + file_name, 'r') as f:
        write_dict = {}
        for line in f:
            word = line.rstrip('\n')
            if word not in embeddings_dict: #dictionaryのキーに着目 key1: val1, 存在しない場合は書き込まず
                print(word + "is not exist")
            else:
                write_dict[word] = embeddings_dict[word]
        file_name = re.sub('.txt', '', file_name)
        pickle_dump(write_dict, '/home/oza/pre-experiment/glove/200d_dic/' + file_name + '.pickle')


