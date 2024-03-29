import pytest
import torch
from data import BERTSimilarity, FlickrDataset
from dataloader import DsetSGPairwise, concat_data
from model import HGAN
from torch.utils.data import DataLoader


@pytest.mark.parametrize('sample_mode', ['tail_random', 'rerank_random'])
def test_HGAN_bbox(sample_mode):
    sim_mat_file = '/data/project/rw/CBIR/data/f30k/f30k_sbert_mean.npy'
    sim_id_file = '/data/project/rw/CBIR/data/f30k/f30k_sbert_img_id.npy'
    sg_path = '/data/project/rw/CBIR/data/f30k/f30k_sgg_freq_prior_with_adj.pkl'
    vocab_emb_path = '/data/project/rw/CBIR/data/f30k/glove_embs_f30k_sgg_freq_prior.pkl' 
    vocab2idx_path = '/data/project/rw/CBIR/data/f30k/vocab2idx_f30k_sgg_freq_prior.pkl' 
    idx2vocab_path = '/data/project/rw/CBIR/data/f30k/idx2vocab_f30k_sgg_freq_prior.pkl'
    sims = BERTSimilarity(sim_mat_file, sim_id_file)
    ds = FlickrDataset(vocab_emb=vocab_emb_path, vocab2idx=vocab2idx_path, idx2vocab=idx2vocab_path,
                       sg_path=sg_path)

    train_dset = DsetSGPairwise(ds, sims, tail_range=100, split='train',
                                mode='fixedbbox', sample_mode=sample_mode,
                                num_steps=3, max_num_he=100)
    train_dloader = DataLoader(train_dset, batch_size=2, num_workers=0,
                               shuffle=True)

    batch = next(iter(train_dloader))
    if sample_mode in ('tail_random', 'rerank_random'):
        anchor, pair, score = batch
        assert anchor.shape == (2, 100, 3, 3, 64, 64)
        assert pair.shape == (2, 100, 3, 3, 64, 64)
    # else sample_mode == 'tmb':
    #     anchor, pair, score = batch
    #     assert anchor.shape == (10, 100, 3, 4)
    #     assert pair.shape == (10, 100, 3, 4)

    cfg = {'MODEL': {'WORD_EMB_SIZE': 512,
                     'NUM_HIDDEN': 100,
                     'NUM_HEAD': 8,
                     'TARGET': 'SBERT',
                     'FEATURE': 'bbox'
                     }}
    model = HGAN(cfg)
    score, att_map = model.score(anchor, pair)
    print(score)


@pytest.mark.parametrize('sample_mode', ['tail_random'])
def test_HGAN_bbox_rel(sample_mode):
    sim_mat_file = '/data/project/rw/CBIR/data/f30k/f30k_sbert_mean.npy'
    sim_id_file = '/data/project/rw/CBIR/data/f30k/f30k_sbert_img_id.npy'
    sg_path = '/data/project/rw/CBIR/data/f30k/f30k_sgg_freq_prior_with_adj.pkl'
    vocab_emb_path = '/data/project/rw/CBIR/data/f30k/glove_embs_f30k_sgg_freq_prior.pkl' 
    vocab2idx_path = '/data/project/rw/CBIR/data/f30k/vocab2idx_f30k_sgg_freq_prior.pkl' 
    idx2vocab_path = '/data/project/rw/CBIR/data/f30k/idx2vocab_f30k_sgg_freq_prior.pkl'
    sims = BERTSimilarity(sim_mat_file, sim_id_file)
    ds = FlickrDataset(vocab_emb=vocab_emb_path, vocab2idx=vocab2idx_path, idx2vocab=idx2vocab_path,
                       sg_path=sg_path)

    train_dset = DsetSGPairwise(ds, sims, tail_range=100, split='train',
                                mode='bbox_word', sample_mode=sample_mode,
                                num_steps=3, max_num_he=100)
    train_dloader = DataLoader(train_dset, batch_size=2, num_workers=0,
                               shuffle=True)

    batch = next(iter(train_dloader))
    if sample_mode == 'tail_random':
        anchor, pair, score = batch
        assert isinstance(anchor, dict)
        assert isinstance(pair, dict)
        assert 'bbox' in anchor
        assert 'word' in anchor
        assert anchor['bbox'].shape == (2, 100, 3, 3, 64, 64)
        assert anchor['word'].shape == (2, 100, 300)
    # else sample_mode == 'tmb':
    #     anchor, pair, score = batch
    #     assert anchor.shape == (10, 100, 3, 4)
    #     assert pair.shape == (10, 100, 3, 4)

    cfg = {'MODEL': {'WORD_EMB_SIZE': 512,
                     'NUM_HIDDEN': 100,
                     'NUM_HEAD': 8,
                     'TARGET': 'SBERT',
                     'FEATURE': 'bbox_word'
                     }}
    model = HGAN(cfg)
    score, att_map = model.score(anchor, pair)
    print(score)


@pytest.mark.parametrize('sample_mode', ['tripletV1', 'rankV1'])
def test_HGAN_triplet(sample_mode):
    sim_mat_file = '/data/project/rw/CBIR/data/f30k/f30k_sbert_mean.npy'
    sim_id_file = '/data/project/rw/CBIR/data/f30k/f30k_sbert_img_id.npy'
    sg_path = '/data/project/rw/CBIR/data/f30k/f30k_sgg_freq_prior_with_adj.pkl'
    vocab_emb_path = '/data/project/rw/CBIR/data/f30k/glove_embs_f30k_sgg_freq_prior.pkl' 
    vocab2idx_path = '/data/project/rw/CBIR/data/f30k/vocab2idx_f30k_sgg_freq_prior.pkl' 
    idx2vocab_path = '/data/project/rw/CBIR/data/f30k/idx2vocab_f30k_sgg_freq_prior.pkl'
    sims = BERTSimilarity(sim_mat_file, sim_id_file)
    ds = FlickrDataset(vocab_emb=vocab_emb_path, vocab2idx=vocab2idx_path, idx2vocab=idx2vocab_path,
                       sg_path=sg_path)

    train_dset = DsetSGPairwise(ds, sims, tail_range=100, split='train',
                                mode='word', sample_mode=sample_mode,
                                num_steps=3, max_num_he=100, pos_k=10)
    train_dloader = DataLoader(train_dset, batch_size=2, num_workers=0,
                               shuffle=True)

    batch = next(iter(train_dloader))
    anchor, pos, neg = batch
    assert isinstance(anchor, torch.Tensor)
    assert isinstance(pos, torch.Tensor)
    assert isinstance(pos, torch.Tensor)
    assert 'bbox' in anchor
    assert 'word' in anchor

    cfg = {'MODEL': {'WORD_EMB_SIZE': 300,
                     'NUM_HIDDEN': 100,
                     'NUM_HEAD': 8,
                     'TARGET': 'margin',
                     'FEATURE': 'word',
                     'LOSS_MARGIN': 0.2
                     }}
    model = HGAN(cfg)
    score, att_map = model.score(anchor, pos)
    print(score)


def test_HAN_GCN():
    sim_mat_file = '/data/project/rw/CBIR/data/f30k/f30k_sbert_mean.npy'
    sim_id_file = '/data/project/rw/CBIR/data/f30k/f30k_sbert_img_id.npy'
    sg_path = '/data/project/rw/CBIR/data/f30k/f30k_sgg_freq_prior_with_adj.pkl'
    vocab_emb_path = '/data/project/rw/CBIR/data/f30k/glove_embs_f30k_sgg_freq_prior.pkl' 
    vocab2idx_path = '/data/project/rw/CBIR/data/f30k/vocab2idx_f30k_sgg_freq_prior.pkl' 
    idx2vocab_path = '/data/project/rw/CBIR/data/f30k/idx2vocab_f30k_sgg_freq_prior.pkl'
    sims = BERTSimilarity(sim_mat_file, sim_id_file)
    ds = FlickrDataset(vocab_emb=vocab_emb_path, vocab2idx=vocab2idx_path, idx2vocab=idx2vocab_path,
                       sg_path=sg_path)

    train_dset = DsetSGPairwise(ds, sims, tail_range=100, split='train',
                                mode='adj_he', sample_mode='tail_random',
                                num_steps=3, max_num_he=100)
    train_dloader = DataLoader(train_dset, batch_size=2, num_workers=0,
                               shuffle=True, collate_fn=concat_data)

    batch = next(iter(train_dloader))
    anchor, pos, score = batch
    assert isinstance(anchor, dict)
    assert isinstance(pos, dict)
    assert 'adj' in anchor
    assert 'x' in anchor
    assert 'n_node' in anchor
    assert 'he' in anchor
    assert anchor['he'].shape == (2, 100, 3)

    cfg = {'MODEL': {'WORD_EMB_SIZE': 300,
                     'NUM_HIDDEN': 100,
                     'NUM_HEAD': 8,
                     'TARGET': 'SBERT',
                     'FEATURE': 'adj_he',
                     }}
    model = HGAN(cfg)
    score, att_map = model.score(anchor, pos)
    print(score)




