import numpy as np
import h5py

fname_in_h5py = '/data/public/rw/datasets/visual_genome/BERT_feature/SBERT_sims.hdf5'
in_h5py = h5py.File(fname_in_h5py, 'r')

vg_ids = list(in_h5py['id'])
mat = np.array([ in_h5py['sims/{}'.format(k)][()] for k in vg_ids ])
print(len(vg_ids))
print(mat.shape)
np.savez('/data/public/rw/datasets/visual_genome/BERT_feature/bert_sim.npz', sims=mat, id=np.array(vg_ids))