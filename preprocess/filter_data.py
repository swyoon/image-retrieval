import sys
sys.path.append("..")
from preprocess.utils import load_files, save_json
import time
import h5py

tic = time.time()
print("data loading")
vg_path = '/data/public/rw/datasets/visual_genome/'
fname_sg = vg_path+'new_scene_graphs.json'
all_sg = load_files(fname_sg)
print("all sg data is loaded, {}s".format(time.time()-tic))

fname_vg_coco_id = vg_path+'BERT_feature/SBERT_sims.hdf5'
h5_vg_coco_id = h5py.File(fname_vg_coco_id, 'r')
vg_coco_id = h5_vg_coco_id['id'][:]

filtered_sg = []

for sg in all_sg:
    if sg['image_id'] in vg_coco_id:
        filtered_sg.append(sg)

save_json(filtered_sg, vg_path+'filtered_scene_graphs_coco.json')
