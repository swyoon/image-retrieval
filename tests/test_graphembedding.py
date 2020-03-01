import torch
from torch.utils.data import DataLoader
import pytest
from dataloader import DsetImgPairwise, DsetSGPairwise, concat_data, repeat_data
from data import BERTSimilarity, FlickrDataset
from model import GraphEmbedding


def test_GraphEmbedding():
    mode = 'adj'
    sims = BERTSimilarity('/data/project/rw/CBIR/data/f30k/f30k_sbert_mean.npy',
                          '/data/project/rw/CBIR/data/f30k/f30k_sbert_img_id.npy')

    vocab_emb_path = '/data/project/rw/CBIR/data/f30k/glove_embs_f30k_sgg_freq_prior.pkl'
    vocab2idx_path = '/data/project/rw/CBIR/data/f30k/vocab2idx_f30k_sgg_freq_prior.pkl'
    idx2vocab_path = '/data/project/rw/CBIR/data/f30k/idx2vocab_f30k_sgg_freq_prior.pkl'
    ds = FlickrDataset(vocab_emb=vocab_emb_path, vocab2idx=vocab2idx_path, idx2vocab=idx2vocab_path,
                       sg_path=None)
    dset = DsetSGPairwise(ds, sims, 3, max_num_he=2, num_steps=3, split='train', mode=mode)
    dl = DataLoader(dset, num_workers=0, batch_size=3, collate_fn=concat_data)
    anchor_data, pair_data, score = next(iter(dl))
    assert isinstance(anchor_data, dict)
    assert isinstance(pair_data, dict)
    # a = concat_data([anchor_data, anchor_data])
    #  b = repeat_data(anchor_data, 2)
    assert 'x' in anchor_data
    assert 'adj' in anchor_data
    assert 'x' in pair_data
    assert 'adj' in pair_data 
    # assert (a['x'] == b['x']).all()
    # assert (a['adj'] == b['adj']).all()


    # c = concat_data([anchor_data, pair_data])
    # print(c)


    cfg = {'MODEL': {'WORD_EMB_SIZE': 300,
                     'NUM_HIDDEN': 100,
                     'NUM_HEAD': 8,
                     'TARGET': 'SBERT',
                     'FEATURE': 'word',
                     'ALGO': 'GCN'
                     }}

    model = GraphEmbedding(cfg)
    score, _ = model.score(anchor_data, pair_data)
    assert len(score) == len(anchor_data['adj'])
