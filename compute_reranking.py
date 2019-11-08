"""
compute_reranking.py
====================
Convert predicted similarity files with Resnet feature reranking
"""
import h5py
import pandas as pd
import sys
import os
import pickle
import argparse
import json
import numpy as np
from time import time
from tqdm import tqdm


def get_visual_genome_coco_split():
    split_file = '/data/project/rw/CBIR/img_split.json'
    with open(split_file, 'r') as f:
        d_id_split = json.load(f)

    l_train = [k for k, v in d_id_split.items() if v == 'train']
    l_test = [k for k, v in d_id_split.items() if v == 'test']
    return l_train, l_test


# setting
parser = argparse.ArgumentParser()
parser.add_argument('algorithm', type=str, help='The name of directory to be searched for predicted similarity score')
args = parser.parse_args()

algorithm = args.algorithm
DB_SET = 'test'
DIST = 'cosine'
n_rerank = 100
result_dir = '/data/project/rw/viewer_CBIR/viewer/results/'
pred_sim_dir = os.path.join(result_dir, algorithm)
output_dir = os.path.join(result_dir, f'{algorithm}_rerank')
print(f'running for {algorithm}')
print(f'reranking {n_rerank}')
print(f'results will be saved at output_dir')

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
    print(f'creating {output_dir}...')

# get resnet feature
resnet_feature_file = '/data/public/rw/datasets/visual_genome/Resnet_feature/wholeImg.hdf5'
f = h5py.File(resnet_feature_file, 'r')
id2idx = {img_id: int(idx) for idx, img_id in enumerate(f['id'])}

l_train, l_test = get_visual_genome_coco_split()
if DB_SET == 'train':
    l_target = l_train[:]
elif DB_SET == 'test':
    l_target = l_test[:]

print('loading resnet feature...')
time_s = time()
l_target_feature = []
for tg_img_id in tqdm(l_target):
    tg_img_idx = id2idx[tg_img_id]
    target_feature = f['data'][tg_img_idx]
    l_target_feature.append(target_feature)
target_features = np.array(l_target_feature)
print(f'{time() - time_s:.2f} sec')

id2tgtidx = {img_id: img_idx for img_idx, img_id in enumerate(l_target)}


# load list of files
l_files = os.listdir(pred_sim_dir)
print(f'processing {len(l_files)} files...')

# for each file
for pred_file in tqdm(l_files):
    query_id = pred_file.split('.')[0]
    img_idx = id2idx[query_id]
    test_feature = f['data'][img_idx]
 
    # read prediction file
    query_sim_file = os.path.join(pred_sim_dir, query_id + '.tsv')
    df_pred_sim = pd.read_csv(query_sim_file, header=None, sep='\t')
    df_pred_sim = df_pred_sim.drop(index=df_pred_sim.index[df_pred_sim[0]==int(query_id)])  # remove self
    # print(df_pred_sim.sort_values(1, ascending=False).head())
    l_candidate_id = list(df_pred_sim[0])
    l_candidate_idx = [id2tgtidx[str(c_id)] for c_id in l_candidate_id]
    candidate_features = target_features[l_candidate_idx]  # 

    # compute resnet feature
    if DIST == 'euclidean':
        l2_dist = np.sqrt(np.sum((test_feature  - candidate_features) ** 2, axis=1))
        sim = - l2_dist
    elif DIST == 'cosine':
        sim = (candidate_features * test_feature).sum(axis=1) / np.sqrt(np.sum(candidate_features ** 2, axis=1)) / np.sqrt(np.sum(test_feature ** 2))
    else:
        raise ValueError(f'{DIST}')
 
    # set their similarity as 0
    df_pred_sim['resnet'] = sim
    df_pred_sim = df_pred_sim.sort_values('resnet', ascending=False)[:n_rerank]
    df_pred_sim[[0, 1]].to_csv(f'/data/project/rw/viewer_CBIR/viewer/results/{algorithm}_rerank/{query_id}.tsv', sep='\t', header=False, index=False)
    # print(df_pred_sim.sort_values(1, ascending=False).head())

