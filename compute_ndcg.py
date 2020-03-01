"""
compute_ndcg.py
===============
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
from data import CocoDataset, FlickrDataset, get_reranked_ids, BERTSimilarity, VGDataset

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

if __name__ == '__main__':

    #
    #  Configuration
    #

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='dataset name', choices=('vg_coco', 'coco', 'f30k'))
    parser.add_argument('model_name', type=str, help='The name of directory to be searched for predicted similarity score')
    parser.add_argument('--resultdir', type=str, default='/data/project/rw/viewer_CBIR/viewer/results/')
    parser.add_argument('--json', action='store_true', help='use separate json file to load query image ids. only valid for vg_coco dataset')
    parser.add_argument('--zero-baseline', action='store_true')
    # parser.add_argument('--resnet', type=float, default=None, help='portion of resnet similarity in relevance')
    args = parser.parse_args()

    if args.model_name == 'random':
        pred_sim_dir = f'/data/project/rw/viewer_CBIR/viewer/{args.dataset}_results/resnet'
    else:
        pred_sim_dir = f'/data/project/rw/viewer_CBIR/viewer/{args.dataset}_results/{args.model_name}'
    # pred_sim_dir = os.path.join(args.resultdir, args.model_name)
    print(f'Computing NDCG for {pred_sim_dir}...')

    if args.dataset == 'coco':
        ds = CocoDataset()
        l_query_id = ds.d_split['test']
        bert_sim_file = '/data/project/rw/CBIR/data/coco/coco_sbert_mean.npy'
        bert_id_file = '/data/project/rw/CBIR/data/coco/coco_sbert_img_id.npy'
    elif args.dataset == 'f30k':
        ds = FlickrDataset()
        l_query_id = ds.d_split['test']
        bert_sim_file = '/data/project/rw/CBIR/data/f30k/f30k_sbert_mean.npy'
        bert_id_file = '/data/project/rw/CBIR/data/f30k/f30k_sbert_img_id.npy'
    elif args.dataset == 'vg_coco':
        ds = VGDataset()
        if args.json:
            selected_id_file = 'test_id_1000_v3.json'
            with open(selected_id_file, 'r') as f:
                l_query_id = json.load(f)
                l_query_id = list(map(int, l_query_id))
        else:
            l_query_id = ds.d_split['test']
        bert_sim_file = '/data/project/rw/CBIR/data/vg_coco/vg_coco_sbert_mean.npy'
        bert_id_file = '/data/project/rw/CBIR/data/vg_coco/vg_coco_sbert_img_id.npy'
    print(f'running for {len(l_query_id)} queries')

    # if args.query == 'all':
    #     l_query_id = [s.split('.')[0] for s in os.listdir(pred_sim_dir)]
    #     print(f'running for all tsv files in {pred_sim_dir}')
    # elif args.query == 'paper':
    #     selected_id_file = 'test_id_1000_v3.json'
    #     # selected_id_file = 'test_id_1000.json'
    #     with open(selected_id_file, 'r') as f:
    #         l_query_id = json.load(f)
    #     print(f'running for selected ids in {selected_id_file}')
    # else:
    #     l_query_id = args.query.strip().split(',')
    #     print(f'running for {l_query_id}')

    if args.zero_baseline:
        print('adjust the smallest similarity score to zero')
        if args.resnet:
            print('using minimum score for resnet weight 0.55')
            min_score = 0.606
            # 새로운 sbert score 는 -1에서 1인 것이 반영됨
        else:
            min_score = 0.59
        print(f'min_score {min_score} ')

        # min_sbert_score = 0.59
        # min_resnet_score = 1.124
        # print(f'min_sbert_score {min_sbert_score} ')
        # print(f'min_resnet_score {min_resnet_score} ')
    else:
        print('Using raw similarity score')

    """load true similarity files"""
    sbert_sim = BERTSimilarity(bert_sim_file, bert_id_file)

    """load train/test split"""
    ks = (5, 10, 20, 30, 40, 50)
    d_result = {k: [] for k in ks}

    # import warnings
    # warnings.filterwarnings("error")

    for query_id in tqdm(l_query_id):
        query_sim_file = os.path.join(pred_sim_dir, f'{query_id}.tsv')
        df_pred_sim = pd.read_csv(query_sim_file, header=None, sep='\t')
        df_pred_sim.columns = ['img_id', 'sim']
        df_pred_sim = df_pred_sim.set_index('img_id')

        # ----- reranking
        # print(len(df_pred_sim))
        l_reranked = get_reranked_ids(args.dataset, query_id)
        df_pred_sim = df_pred_sim.loc[l_reranked]
        # print(len(df_pred_sim))

        # df_pred_sim = df_pred_sim.drop(index=df_pred_sim.index[df_pred_sim.index==int(query_id)])  # drop self
        l_candidate_id = list(df_pred_sim.index)

        true_sim = torch.tensor([sbert_sim.get_similarity(query_id, img_id) for img_id in l_candidate_id]).view(1, -1).to(dtype=torch.float)
        true_sim.clamp_(min=0)
        # if args.zero_baseline:
        #     true_sim -= min_score
        if args.model_name == 'random':
            pred_sim = torch.rand_like(true_sim)
        else:
            pred_sim = torch.tensor(df_pred_sim['sim'].values).view(1, -1)

        assert torch.all(true_sim >= 0)
        d_ndcg = ndcg_batch(pred_sim, true_sim, ks=ks)
        # d_ndcg = ndcg_batch(-true_sim, true_sim, ks=ks)
        for k in ks:
            d_result[k].append(d_ndcg[k])


    d_result = {k: np.mean(v) for k, v in d_result.items()} 
    for k in sorted(d_result.keys()):
        print(k, f'{d_result[k]:.4f}')
    print(','.join(map(str, ks)))
    print(','.join([f'{d_result[k]:.4f}' for k in ks]))
    # save result
    # output_file = f"ndcg_{args.model_name}.pkl"
    # with open(output_file, "wb") as f:
    #     pickle.dump(d_result, f)
    #     print(f'result saved in {output_file}')

