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
from data import BERTSimilarity, get_karpathy_split_light, FlickrDataset, VGDataset


# DATASET = 'f30k'
# DB_SET = 'train'  # one of ('train', 'val', 'test')
DATASET = sys.argv[1]
DB_SET = sys.argv[2]
split_idx = int(sys.argv[3])
n_split = int(sys.argv[4])
DIST = 'cosine'  # one of ('euclidean', 'cosine')
print(f'DATASET : {DATASET}')
print('Preparing similarity scores for CBIR web viewer...')
print('Running for resnet')
print(f'Database set: {DB_SET}')
print(f'Using similarity metric {DIST}')

print('Preparing similarity scores for CBIR web viewer...')
print(f'split {split_idx} out of  {n_split}')

if DATASET == 'coco':
    # time_s = time()
    # sims = BERTSimilarity('/data/project/rw/CBIR/coco_f30k/coco_sbert_mean.npy',
    #                '/data/project/rw/CBIR/coco_f30k/coco_sbert_img_id.npy')
    # print(f'Loading SBERT score {time() - time_s:.2f} sec')

    d_split = get_karpathy_split_light()
    l_test = d_split[DB_SET]
    resnet_feature_file = '/data/project/rw/CBIR/data/coco/resnet152.h5'
elif DATASET == 'f30k':
    f30k = FlickrDataset()
    l_test = f30k.d_split[DB_SET]
    resnet_feature_file = '/data/project/rw/CBIR/data/f30k/resnet152.h5'
elif DATASET == 'vg_coco':
    vg = VGDataset()
    l_test = vg.d_split[DB_SET]
    resnet_feature_file = '/data/project/rw/CBIR/data/vg_coco/resnet152.h5'
elif DATASET == 'vg_coco_sp':
    vg = VGDataset(new_split=True)
    l_test = vg.d_split[DB_SET]
    resnet_feature_file = '/data/project/rw/CBIR/data/vg_coco/resnet152.h5'

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

if DB_SET == 'test':
    out_dir = f'/data/project/rw/viewer_CBIR/viewer/{DATASET}_results/resnet'
elif DB_SET == 'train':
    out_dir = f'/data/project/rw/viewer_CBIR/viewer/{DATASET}_results/train_resnet'
elif DB_SET == 'val':
    out_dir = f'/data/project/rw/viewer_CBIR/viewer/{DATASET}_results/val_resnet'
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)


'''compute pair-wise similarity'''
def split_id(l, split_idx, n_split):
    split_size = int(len(l) / n_split)
    if split_idx == n_split - 1:
        end = len(l)
    else:
        end = (split_idx + 1) * split_size
    start = split_idx * split_size
    print(start, end)
    return l[start:end]


time_s = time()
l_test = split_id(l_test, split_idx, n_split)
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
