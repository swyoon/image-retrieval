import sys
sys.path.append("..")
from utils import load_files, save_pickle, clean_str
import time
import numpy as np
import h5py

vocab2idx = {"<PAD>":0, "<UNK>":1, "<START>":2, "<END>":3}
vocab_objects = []
vocab_relations = []
vocab_attributes = []

idx2vocab = ["<PAD>",  "<UNK>", "<START>", "<END>"]
init_len = len(idx2vocab)
vocab_len = init_len
not_in = 0

print("load: glove")
glove_path = '/data/project/rw/VisualGenome/glove_data/glove.6B.300d.pkl'
glove = load_files(glove_path)
print("loaded: glove")

print("load: sg")
tic = time.time()
fname_sg = '/data/public/rw/datasets/visual_genome/filtered_scene_graphs_coco.json'
sgs = load_files(fname_sg)
print("all sg data is loaded, {}s".format(time.time()-tic))

for i, sg in enumerate(sgs):
    # if i > 1000:
    #     break
    if i % 1000 == 0:
        print("{}/{}".format(i, len(sgs)))

    objects = sg['objects']
    relations = sg['relationships']

    for obj in objects:
        names = obj['names']
        for name in names:
            name = clean_str(name)
            if name not in vocab2idx:
                vocab2idx[name] = vocab_len
                idx2vocab.append(name)
                vocab_len += 1

            if name not in vocab_objects:
                vocab_objects.append(name)

        if 'attributes' in obj.keys():
            atts = obj['attributes']
            for att in atts:
                att = clean_str(att)
                if att not in vocab2idx:
                    vocab2idx[att] = vocab_len
                    idx2vocab.append(att)
                    vocab_len += 1
                if att not in vocab_attributes:
                    vocab_attributes.append(att)

    for rel in relations:
        predicate = clean_str(rel['predicate'])
        if predicate not in vocab2idx:
            vocab2idx[predicate] = vocab_len
            idx2vocab.append(predicate)
            vocab_len += 1

        if predicate not in vocab_relations:
            vocab_relations.append(predicate)

print("total vocab number: {}".format(vocab_len))

glove_embs = np.zeros([vocab_len, 300])
for idx, strs in enumerate(idx2vocab[init_len:]):
    tokens = strs.strip().split(' ')
    if len(tokens) > 1:
        glove_for_edge = []
        for token in tokens:
            if token in glove:
                glove_for_edge.append(glove[token])

        glove_for_edge = np.vstack(glove_for_edge)
        glove_embs[idx+init_len] = np.mean(glove_for_edge, axis=0, keepdims=True)

    else:
        if strs in glove:
            glove_embs[idx+init_len] = glove[strs]
        else:
            #glove_embs[idx+init_len] = np.random.random([300])
            if not_in < 100 and not_in % 10 == 0:
                print(strs + " Not in glove")
            not_in += 1
print("not in words", not_in)

project_pwd = '/data/project/rw/VisualGenome/CBIR/'
save_pickle(glove_embs, project_pwd+'glove_embs_vg_coco_sg.pkl')
save_pickle(vocab2idx, project_pwd+'vocab2idx_vg_coco_sg.pkl')
save_pickle(idx2vocab, project_pwd+'idx2vocab_vg_coco_sg.pkl')
save_pickle(vocab_objects, project_pwd+'vocab_objects.pkl')
save_pickle(vocab_attributes, project_pwd+'vocab_attributes.pkl')
save_pickle(vocab_relations, project_pwd+'vocab_relations.pkl')