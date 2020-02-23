from dataloader import DsetImgPairwise, DsetSGPairwise
from data import BERTSimilarity

def test_DsetPairwise():
    sims = BERTSimilarity('/data/project/rw/CBIR/data/f30k/f30k_sbert_mean.npy',
                          '/data/project/rw/CBIR/data/f30k/f30k_sbert_img_id.npy')
    dset = DsetImgPairwise('f30k', sims, 3, split='train')
    dset[2]


def test_DsetSGPairwise():
    sims = BERTSimilarity('/data/project/rw/CBIR/data/f30k/f30k_sbert_mean.npy',
                          '/data/project/rw/CBIR/data/f30k/f30k_sbert_img_id.npy')
    dset = DsetSGPairwise('f30k', sims, 3, max_num_he=2, num_steps=3, split='train')
    anchor_data, pair_data, score = dset[2]
    assert anchor_data.shape == (2, 512)
    assert pair_data.shape == (2, 512)
    assert -1 <= score <= 1
