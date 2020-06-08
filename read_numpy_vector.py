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
 

embeddings_dict = read_glove()

test = np.load('/home/oza/pre-experiment/glove/numpy_vector/200d_wiki.npy')
print(test.shape)
print(np.all(test[0] == embeddings_dict['ammo']))

