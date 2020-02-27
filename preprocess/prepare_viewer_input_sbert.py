"""
prepare_viewer_input_sbert.py
=============================
Load SBERT similarity file, produce prediction files for viewer anv NDCG computation
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
import sys
sys.path.append('../')
from data import BERTSimilarity, get_karpathy_split_light, FlickrDataset, VGDataset


DATASET = 'vg_coco'  # one of ('coco', 'f30k')
GENCAP = True
print('Preparing similarity scores for CBIR web viewer...')

if DATASET == 'coco':
    time_s = time()
    sims = BERTSimilarity('/data/project/rw/CBIR/data/coco/coco_sbert_mean.npy',
                   '/data/project/rw/CBIR/coco_f30k/coco_sbert_img_id.npy')
    print(f'Loading SBERT score {time() - time_s:.2f} sec')

    d_split = get_karpathy_split_light()
    l_test = d_split['test']

elif DATASET == 'f30k':
    time_s = time()
    sims = BERTSimilarity('/data/project/rw/CBIR/data/f30k/f30k_sbert_mean.npy',
                          '/data/project/rw/CBIR/data/f30k/f30k_sbert_img_id.npy')
    print(f'Loading SBERT score {time() - time_s:.2f} sec')

    flickr = FlickrDataset()
    d_split = flickr.d_split
    l_test = d_split['test']
    print(f'total {len(l_test)} images')
elif DATASET == 'vg_coco':
    vg = VGDataset()
    l_test = vg.d_split['test']
    time_s = time()
    if GENCAP:
        sims = BERTSimilarity('/data/project/rw/CBIR/data/vg_coco/vg_coco_gencap_sbert.npy',
                              '/data/project/rw/CBIR/data/vg_coco/vg_coco_gencap_sbert_img_id.npy')
        print('using sbert score for generated captions')
    else:
        sims = BERTSimilarity('/data/project/rw/CBIR/data/vg_coco/vg_coco_sbert_mean.npy',
                              '/data/project/rw/CBIR/data/vg_coco/vg_coco_sbert_img_id.npy')
    print(f'Loading SBERT score {time() - time_s:.2f} sec')
else:
    raise ValueError


if GENCAP:
    result_dir = f'/data/project/rw/viewer_CBIR/viewer/{DATASET}_results/gencap_sbert/'
else:
    result_dir = f'/data/project/rw/viewer_CBIR/viewer/{DATASET}_results/sbert/'

if not os.path.isdir(result_dir):
    os.mkdir(result_dir)

for test_id in tqdm(l_test):  # query id
    l_sim = []
    for target_id in l_test:
#         train_idx = sbert_id2idx[train_id]
#         print(f['sims'][test_id][train_idx])
#         l_sim.append(f['sims'][test_id][train_idx])
        l_sim.append(sims.get_similarity(test_id, target_id))
    df = pd.DataFrame({'target_id': l_test, 'sim': l_sim}).sort_values('target_id')

    df[['target_id', 'sim']].to_csv(os.path.join(result_dir, f'{test_id}.tsv'), sep='\t', header=False, index=False)
