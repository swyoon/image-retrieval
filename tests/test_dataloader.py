import torch
from dataloader import DsetImgPairwise, DsetSGPairwise, concat_data, repeat_data
from data import BERTSimilarity, FlickrDataset
import pytest

@pytest.mark.parametrize('dataset', ['f30k', 'vg_coco'])
def test_DsetPairwise(dataset):
    sims = BERTSimilarity(f'/data/project/rw/CBIR/data/{dataset}/{dataset}_sbert_mean.npy',
                          f'/data/project/rw/CBIR/data/{dataset}/{dataset}_sbert_img_id.npy')
    dset = DsetImgPairwise(dataset, sims, 3, split='train')
    dset[2]


def test_DsetSGPairwise_word():
    mode = word
    sims = BERTSimilarity('/data/project/rw/CBIR/data/f30k/f30k_sbert_mean.npy',
                          '/data/project/rw/CBIR/data/f30k/f30k_sbert_img_id.npy')

    vocab_emb_path = '/data/project/rw/CBIR/data/f30k/glove_embs_f30k_sgg_freq_prior.pkl'
    vocab2idx_path = '/data/project/rw/CBIR/data/f30k/vocab2idx_f30k_sgg_freq_prior.pkl'
    idx2vocab_path = '/data/project/rw/CBIR/data/f30k/idx2vocab_f30k_sgg_freq_prior.pkl'
    ds = FlickrDataset(vocab_emb=vocab_emb_path, vocab2idx=vocab2idx_path, idx2vocab=idx2vocab_path,
                       sg_path=None)
    dset = DsetSGPairwise(ds, sims, 3, max_num_he=2, num_steps=3, split='train', mode=mode)
    anchor_data, pair_data, score = dset[2]
    assert anchor_data.shape == (2, 300)
    assert pair_data.shape == (2, 300)
    assert -1 <= score <= 1
