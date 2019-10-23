from utils import load_files, save_pickle, clean_str
import time
import numpy as np

print("load: sg")
tic = time.time()
fname_sg = '/data/public/rw/datasets/visual_genome/filtered_scene_graphs_coco.json'
sgs = load_files(fname_sg)
print("all sg data is loaded, {}s".format(time.time()-tic))

fname_split_info = "/data/project/rw/CBIR/img_split.json"
split_info = load_files(fname_split_info)

new_sgs_train = {}
new_sgs_test = {}
num_sg_with_no_graph = 0

for i, sg in enumerate(sgs):
    new_sg = {}
    vg_image_id = str(sg['image_id'])

    if vg_image_id not in split_info.keys():
        continue

    if i % 1000 == 0:
        print("{}/{}".format(i, len(sgs)))

    objects = sg['objects']
    relations = sg['relationships']

    num_nodes = 0
    nodes = []
    bboxes = []
    nodes_id = []

    for obj in objects:
        num_nodes += 1
        if 'attributes' in obj.keys():
            num_nodes += len(obj['attributes'])

    num_nodes += len(relations)
    if num_nodes == 0:
        print("there is no nodes, img_id: {}".format(vg_image_id))
        num_sg_with_no_graph += 1
        continue

    adj = np.zeros([num_nodes, num_nodes], dtype=np.uint8)

    for obj in objects:
        name_obj = clean_str(obj['names'][0])
        nodes.append(name_obj)
        obj_idx = len(nodes)-1
        nodes_id.append( obj['object_id'] )

        if 'attributes' in obj.keys():
            for att in obj['attributes']:
                name_att = clean_str(att)
                num_nodes += 1
                nodes.append(name_att)
                att_idx = len(nodes)-1
                adj[att_idx, obj_idx] = 1

    for rel in relations:
        predicate = clean_str(rel['predicate'])
        nodes.append(predicate)
        pred_idx = len(nodes)-1

        sub_id = rel['subject_id']
        sub_idx = nodes_id.index(sub_id)

        obj_id = rel['object_id']
        obj_idx = nodes_id.index(obj_id)

        adj[sub_idx, pred_idx] = 1
        adj[pred_idx, obj_idx] = 1

    new_sg['node_labels'] = nodes
    new_sg['adj'] = adj

    if split_info[vg_image_id] == 'train':
        new_sgs_train[vg_image_id] = new_sg
    elif split_info[vg_image_id] == 'test':
        new_sgs_test[vg_image_id] = new_sg

print("number of train sg: {}".format(len(new_sgs_train)))
print("number of test sg: {}".format(len(new_sgs_test)))

fname_new_sg_train = '/data/project/rw/CBIR/scene_graph_with_adj_train.pkl'
fname_new_sg_test = '/data/project/rw/CBIR/scene_graph_with_adj_test.pkl'
save_pickle(new_sgs_train, fname_new_sg_train)
save_pickle(new_sgs_test, fname_new_sg_test)