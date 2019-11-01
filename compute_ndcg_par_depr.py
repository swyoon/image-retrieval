"""
compute_ndcg.py
===============
Parallelized version. Not very efficient.
Computes Normalized Discounted Cumulative Gain (NDCG)
"""
import numpy as np
import torch
import h5py
import os, json, pickle
import sys
import pandas as pd
from collections import defaultdict
import time
import argparse
from tqdm import tqdm
from joblib import Parallel, delayed

#
#  Configuration
#

parser = argparse.ArgumentParser()
parser.add_argument('model_name', type=str, help='The name of directory to be searched for predicted similarity score')
parser.add_argument('--resultdir', type=str, default='/data/project/rw/viewer_CBIR/viewer/results/')
parser.add_argument('--query', type=str, default='all', help='list of query ids. comma separated.')
parser.add_argument('--par', type=int, default=6, help='The number of workers for parallelization')
args = parser.parse_args()

pred_sim_dir = os.path.join(args.resultdir, args.model_name)
print(f'Computing NDCG for {pred_sim_dir}...')

if args.query == 'all':
    l_query_id = [s.split('.')[0] for s in os.listdir(pred_sim_dir)]
    print(f'running for all tsv files in {pred_sim_dir}')
else:
    l_query_id = args.query.strip().split(',')
    print(f'running for {l_query_id}')
print(f'running for {len(l_query_id)} queries')
print(f'using {args.par} workders')

#
# Functions
#

def dcg_batch(sims, labels, ks=(5, 10, 20, 30, 40, 50)):
    """
    sims and labels must have a same order
    sims: float, (batch_size, n_g), predicted similarity scores(prediction)
    labels:  true relevance scores
    ks: list of int, top-k
    """

    # sort result
    # print("start sorting - {}".format(time.ctime()))
    max_k = max(ks)
    _, topk_idx = torch.topk(sims, max_k)
    query_idx = torch.arange(sims.size(0)).view(-1, 1).expand(sims.size(0), max_k)
    labels = labels[query_idx, topk_idx].to(dtype=torch.float)
    # print("end sorting - {}".format(time.ctime()))

    result = {}
    for k in ks:
        # 1,2등에 점수 차이를 줄때, log(rank+1)
        #     disc_factor = torch.log2(torch.arange(k, dtype=torch.float)+2) # 1, 2등 차이 존재
        # 1,2등 점수 차이가 같고, log(rank) -- 1등과 2등이 같음
        disc_factor = torch.log2(torch.arange(k, dtype=torch.float) + 1)  # 1, 2등을 같게
        disc_factor[0] = 1

        # for batch
        disc_factor = disc_factor.repeat(sims.size(0), 1)

        dcg_val = (labels[:, :k] / disc_factor).sum(dim=1)
        result[k] = dcg_val

    return result


def ndcg_batch(sims, labels, ks=(5, 10, 20, 30, 40, 50)):
    """
    sims and labels must have a same order
    sims: float, (n_g), predicted similarity scores
    labels: float, (n_g), true relevance scores
    k: int, top-k
    """
    labels = labels.to(torch.float32)
    # print("dcg")
    val = dcg_batch(sims, labels, ks)
    # print("idcg")
    idcg = dcg_batch(labels, labels, ks)

    ndcg = {}
    for k in ks:
        idx = idcg[k] == 0
        res = val[k] / idcg[k]
        res[idx] = 0
        ndcg[k] = res

    return ndcg


def dcg(sims, labels, k=5):
    """
    sims and labels must have a same order
    sims: float, (n_g), predicted similarity scores(prediction)
    labels: int64, (n_g), {0(negative), 1(positive)}^n_g
    k: int, top-k
    """

    k = sims.size(0) if k < 1 or sims.size(0) < k else k
    _, idx = torch.topk(sims, k)
    labels = labels[idx].to(dtype=torch.float)

    # 1,2등에 점수 차이를 줄때, log(rank+1)
    #     disc_factor = torch.log2(torch.arange(k, dtype=torch.float)+2) # 1, 2등 차이 존재
    
    # 1,2등 점수 차이가 같고, log(rank) -- 1등과 2등이 같음
    disc_factor = torch.log2(torch.arange(k, dtype=torch.float) + 1)  # 1, 2등을 같게
    disc_factor[0] = 1
    
    dcg_val = (labels / disc_factor).sum()

    return dcg_val


def ndcg(sims, labels, k=5):
    """
    sims and labels must have a same order
    sims: float, (n_g), predicted similarity scores
    labels: float, (n_g), true relevance scores
    k: int, top-k
    """
    val = dcg(sims, labels, k=k)
    idcg = dcg(labels, labels, k=k)
    ndcg = 0 if idcg == 0 else val / idcg

    return ndcg


def get_train_test_split(split_json='/data/project/rw/CBIR/img_split.json'):
    """return split ids"""
    with open(split_json, 'r') as f:
        split = json.load(f)
	
    l_test_id = []
    l_train_id = []
    for vg_id, val in split.items():
        if val == 'train': # train, train_all 쓰는 것이 맞는가?
            l_train_id.append(vg_id)  # img_split.json과 다르게 int로 바꿔서 저장하여서, 혹시 편하신대로 str쓰셔도 됩니다.

        elif val == 'test':
            l_test_id.append(vg_id)
    return sorted(l_train_id), sorted(l_test_id)


class BERTSimilarityInMem:
    """class for handling SBERT similarity file from IMLAB
    Identical to BERTSimilarity, but loads all similarity metric in memory"""
    def __init__(self, file_path):
        with h5py.File(file_path, 'r') as f:
            self.idx_lookup = {str(img_id): idx for idx, img_id in enumerate(f['id'])}
            self.sims = {}
            for k, v in tqdm(f['sims'].items()):
                self.sims[k] = v[:]
            # self.sims = {k: v[:] for k, v in f['sims'].items()}

    def get_similarity(self, img_id_1, img_id_2):
        img_idx_2 = self.idx_lookup[str(img_id_2)]
        return self.sims[str(img_id_1)][img_idx_2]


def compute(query_id, sim_array):
    """main function for computing NDCG"""
    # d_result = {k: [] for k in ks}
    d_result = {}
    time_1 = time.time()
    query_sim_file = os.path.join(pred_sim_dir, query_id + '.tsv')
    df_pred_sim = pd.read_csv(query_sim_file, header=None, sep='\t')
    time_2 = time.time()
    # print(f'1 {time_2 - time_1}')

    l_candidate_idx = np.array([id2idx[str(img_id)] for img_id in df_pred_sim[0]])
    # l_candidate_id = list(df_pred_sim[0])

    time_3 = time.time()
    # print(f'2 {time_3 - time_2}')
    # with h5py.File("/data/public/rw/datasets/visual_genome/BERT_feature/SBERT_sims_float16.hdf5", "r") as f_sbert:
    #     true_sim = torch.tensor([f_sbert[f'/sims/{query_id}'][img_idx] for img_idx in l_candidate_idx]).view(1, -1)
    # true_sim = torch.tensor([sbert_sim.get_similarity(query_id, img_id) for img_id in l_candidate_id]).view(1, -1)
    true_sim = torch.tensor([sim_array[img_idx] for img_idx in l_candidate_idx]).view(1, -1)
    pred_sim = torch.tensor(df_pred_sim[1].values).view(1, -1)

    time_4 = time.time()
    # print(f'3 {time_4 - time_3}')
    d_ndcg = ndcg_batch(pred_sim, true_sim, ks=ks)
    return d_ndcg
    # for k in ks:
    #     # d_result[k].append(d_ndcg[k])
    #     d_result[k] = d_ndcg[k]
    # time_5 = time.time()
    # # print(f'4 {time_5 - time_4}')
    # return d_result



"""load true similarity files"""
# f_sbert = h5py.File("/data/public/rw/datasets/visual_genome/BERT_feature/SBERT_sims.hdf5", "r")
f_sbert = h5py.File("/data/public/rw/datasets/visual_genome/BERT_feature/SBERT_sims_float16.hdf5", "r")
# f_sbert_gen = h5py.File("/data/public/rw/datasets/visual_genome/BERT_feature/SBERT_gen_sims_new.hdf5", "r")
# f_resnet = h5py.File("/data/public/rw/datasets/visual_genome/Resnet_feature/resnet_coco_cosine_top100.hdf5", "r")
ids = f_sbert['id'][:]
id2idx = {str(img_id): idx for idx, img_id in enumerate(ids)}  
sbert_sim = BERTSimilarityInMem('/data/public/rw/datasets/visual_genome/BERT_feature/SBERT_sims_float16.hdf5')


"""load train/test split"""
l_train_id, l_test_id = get_train_test_split()
ks = (5, 10, 20, 30, 40, 50)
d_result = {k: [] for k in ks}

l_result = Parallel(n_jobs=args.par)(delayed(compute)(query_id, sbert_sim.sims[query_id][:]) for query_id in tqdm(l_query_id))
for k in ks:
    for r in l_result:
        d_result[k].append(r[k])


d_result = {k: np.mean(v) for k, v in d_result.items()} 
print(d_result)
# save result
output_file = f"ndcg_{args.model_name}.pkl"
with open(output_file, "wb") as f:
    pickle.dump(d_result, f)
    print(f'result saved in {output_file}')

