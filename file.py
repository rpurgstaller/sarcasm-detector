import os

import jsonpickle
import json
import csv
import numpy as np
import codecs

from scipy.sparse._matrix_io import save_npz, load_npz

PATH_OUTPUT = os.path.join(os.path.dirname(__file__), "output")

DIR_RAW_DATA = os.path.join(PATH_OUTPUT, "raw_data")
DIR_VECTORIZED_DATA = os.path.join(PATH_OUTPUT, "vectorized_data")
DIR_RESULTS = os.path.join(PATH_OUTPUT, "results")

FILE_DATA_DOCS = "texts.json"

FILE_DATA_X_TRAIN = "X_train.npz"
FILE_DATA_X_TEST = "X_test.npz"
FILE_DATA_Y_TRAIN = "y_train.npy"
FILE_DATA_Y_TEST = "y_test.npy"

FILE_DATA_VOC = "feature_names.json"
FILE_DATA_TARGETS = "targets.json"

def create_if_not_exists(filepath):
    if not os.path.exists(filepath):
        os.makedirs(filepath)
        
def save(path, filename, data):
    
    create_if_not_exists(path)

    with open(os.path.join(path, filename), "w") as f:
        f.write(data)

def load(filepath):
    with open(filepath, "r") as f:
        return f.read()

def save_json_pickle(path, filename, data):
    
    save_json(path, filename, jsonpickle.encode(data))

def load_json_pickle(filepath):
    json = load_json(filepath)
    
    return jsonpickle.decode(json)

def save_json_unicode(path, filename, data):
    with codecs.open(os.path.join(path, filename), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

def save_json(path, filename, data):
    
    create_if_not_exists(path)
    
    with open(os.path.join(path, filename), "w") as f:
        json.dump(data, f)
                
def load_json(filepath):
    with open(filepath) as f:
        return json.load(f)
            
def load_json_unicode(filepath):
    d = json.load(codecs.open(filepath, 'r', 'utf-8-sig'))
    return d

def save_sparse(path, filename, sparse_matrix): 
    
    create_if_not_exists(path)
    
    save_npz(os.path.join(path, filename), sparse_matrix)

def load_sparse(filepath):
    return load_npz(filepath)

def save_array(path, filename, a):
    
    create_if_not_exists(path)
    
    np.save(os.path.join(path, filename), a)
    
def load_array(filepath):
    return np.load(filepath)    
        


