"""
prepare_viewr_input.py
======================
Generate predicted similarity scores for web-based viewer
"""
import h5py
import pandas as pd
import sys
import os
import pickle
import json
import numpy as np
from time import time
from tqdm import tqdm
from joblib import Parallel, delayed
sys.path.append('../')
from data import BERTSimilarity, get_karpathy_split_light, FlickrDataset


DATASET = 'f30k'
DB_SET = 'test'  # one of ('train', 'test')
DIST = 'cosine'  # one of ('euclidean', 'cosine')
print('Preparing similarity scores for CBIR web viewer...')
print('Running for resnet')
print(f'Database set: {DB_SET}')
print(f'Using similarity metric {DIST}')

print('Preparing similarity scores for CBIR web viewer...')

if DATASET == 'coco':
    # time_s = time()
    # sims = BERTSimilarity('/data/project/rw/CBIR/coco_f30k/coco_sbert_mean.npy',
    #                '/data/project/rw/CBIR/coco_f30k/coco_sbert_img_id.npy')
    # print(f'Loading SBERT score {time() - time_s:.2f} sec')

    d_split = get_karpathy_split_light()
    l_test = d_split['test']
    resnet_feature_file = '/data/project/rw/CBIR/data/coco/resnet152.h5'
elif DATASET == 'f30k':
    f30k = FlickrDataset()
    l_test = f30k.d_split['test']
    resnet_feature_file = '/data/project/rw/CBIR/data/f30k/resnet152.h5'


f = h5py.File(resnet_feature_file, 'r')
id2idx = {img_id: int(idx) for idx, img_id in enumerate(f['id'])}


'''load training data array into memory'''
time_s = time()
l_target = l_test[:]

l_target_feature = []
for tg_img_id in tqdm(l_target):
    tg_img_idx = id2idx[tg_img_id]
    target_feature = f['resnet_feature'][tg_img_idx]
    l_target_feature.append(target_feature)
target_features = np.array(l_target_feature)
print(f'{time() - time_s:.2f} sec')

out_dir = f'/data/project/rw/viewer_CBIR/viewer/{DATASET}_results/resnet'
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)


'''compute pair-wise similarity'''
time_s = time()
for img_id in tqdm(l_test):
    result = {'target_id': [], 'sim': []}
    img_idx = id2idx[img_id]
    test_feature = f['resnet_feature'][img_idx]
    
    if DIST == 'euclidean':
        l2_dist = np.sqrt(np.sum((test_feature  - target_features) ** 2, axis=1))
        sim = - l2_dist
    elif DIST == 'cosine':
        sim = (target_features * test_feature).sum(axis=1) / np.sqrt(np.sum(target_features ** 2, axis=1)) / np.sqrt(np.sum(test_feature ** 2))
    else:
        raise ValueError(f'{DIST}')

    # result['query_id'] += [img_id] * len(l_train)
    result['target_id'] = l_target
    result['sim'] = list(sim)
    result = pd.DataFrame(result).sort_values('target_id')
    # if DIST == 'euclidean':
    #     result[['target_id', 'sim']].to_csv(f'/data/project/rw/viewer_CBIR/viewer/results/SET}_resnet/{img_id}.tsv', sep='\t', header=False, index=False)
    # elif DIST == 'cosine':
    result[['target_id', 'sim']].to_csv(os.path.join(out_dir, f'{img_id}.tsv'), sep='\t', header=False, index=False)

print(f'{time() - time_s:.2f} sec')
