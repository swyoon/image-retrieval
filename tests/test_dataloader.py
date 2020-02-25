from dataloader import DsetImgPairwise, DsetSGPairwise
from data import BERTSimilarity, FlickrDataset
import pytest

@pytest.mark.parametrize('dataset', ['f30k', 'vg_coco'])
def test_DsetPairwise(dataset):
    sims = BERTSimilarity(f'/data/project/rw/CBIR/data/{dataset}/{dataset}_sbert_mean.npy',
                          f'/data/project/rw/CBIR/data/{dataset}/{dataset}_sbert_img_id.npy')
    dset = DsetImgPairwise(dataset, sims, 3, split='train')
    dset[2]


def test_DsetSGPairwise():
    sims = BERTSimilarity('/data/project/rw/CBIR/data/f30k/f30k_sbert_mean.npy',
                          '/data/project/rw/CBIR/data/f30k/f30k_sbert_img_id.npy')
    ds = FlickrDataset(sg_path=None)
    dset = DsetSGPairwise(ds, sims, 3, max_num_he=2, num_steps=3, split='train')
    anchor_data, pair_data, score = dset[2]
    assert anchor_data.shape == (2, 512)
    assert pair_data.shape == (2, 512)
    assert -1 <= score <= 1
