import pickle
import json
import yaml

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
    return data

def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)