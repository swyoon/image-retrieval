import numpy as np
import h5py
import json

fname_in_h5py = '/data/public/rw/datasets/visual_genome/BERT_feature/SBERT_sims.hdf5'
fname_img_split = '/data/project/rw/CBIR/img_split.json'

in_h5py = h5py.File(fname_in_h5py, 'r')
img_split = json.load(open(fname_img_split, 'r'))
train_id_list = []
train_idx_list = []
test_id_list = []
test_idx_list = []

vg_ids = list(in_h5py['id'])
id2idx = {val: i for i, val in enumerate(vg_ids)}

for vg_id in vg_ids:
    if str(vg_id) not in img_split.keys():
        continue

    if img_split[str(vg_id)] == 'train':
        train_id_list.append(vg_id)
        train_idx_list.append( id2idx[vg_id] )

    elif img_split[str(vg_id)] == 'test':
        test_id_list.append(vg_id)
        test_idx_list.append( id2idx[vg_id] )
    else:
        print("not valid split, {}".format(img_split[str(vg_id)]))
        break

sim_mat_train_to_all = np.array([in_h5py['sims/{}'.format(k)][train_idx_list] for k in train_id_list])
print(sim_mat_train_to_all.shape)
sim_mat_test_to_all = np.array([in_h5py['sims/{}'.format(k)][()] for k in test_id_list])
print(sim_mat_test_to_all.shape)
"""
mat = np.array([ in_h5py['sims/{}'.format(k)][()] for k in vg_ids ])
print(len(vg_ids))
print(mat.shape)
np.savez('/data/public/rw/datasets/visual_genome/BERT_feature/bert_sim.npz', sims=mat, id=np.array(vg_ids))
"""