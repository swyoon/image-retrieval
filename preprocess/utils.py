import pickle
import json
import yaml
import h5py
import numpy as np
import re


def load_files(path):
    if path.rsplit('.', 2)[-1] == 'json':
        with open(path, 'r') as f:
            data = json.load(f)
    elif path.rsplit('.', 2)[-1] in ['pkl', 'pickle']:
        with open(path, 'rb') as f:
            data = pickle.load(f)
    elif path.rsplit('.', 2)[-1] == 'yaml':
        with open(path, 'r') as f:
            try:
                data = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(exc)
    elif path.rsplit('.', 2)[-1] == 'hdf5':
        data = h5py.File(path, "r")
    elif path.rsplit('.', 2)[-1] == 'npz':
        data = np.load(path)

    return data


def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)


def clean_str(string, lower = True):
    string = re.sub(r"[^A-Za-z0-9,!\']", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'m", " \'m", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    return string.strip().lower() if lower else string.strip()