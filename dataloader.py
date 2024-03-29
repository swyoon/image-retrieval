import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as TF
from torch.utils.data import Dataset
import operator
import os
from data import CocoDataset, FlickrDataset, VGDataset, get_reranked_ids
from PIL import Image

def sampling(prob):
    return np.random.choice(len(prob), 1, p=prob)


def he_sampling(adj, word_vec, num_steps, max_num_he, mode, eps_prob=0.001, word_emb_size=300):
    # word_vec: N x 300 word embeddings

    n_nodes = adj.shape[0]

    num_outedges = np.sum(adj, axis=1) + 0.5
    init_prob = num_outedges / np.sum(num_outedges, keepdims=True)

    adj = adj + eps_prob

    row_sum = np.sum(adj, axis=1, keepdims=True)
    adj = adj / row_sum

    start_node = np.random.choice(n_nodes, max_num_he*2, p=init_prob)

    HEs = [[n] for n in start_node]
    for k in range(num_steps - 1):
        [HE.append(sampling(adj[HE[-1]])[0]) for HE in HEs]

    unique_HEs = []
    for HE in HEs:
        unique_HEs.append(list(np.unique(HE)))
    unique_HEs = list(np.unique(unique_HEs))

    num_HE = min(max_num_he, len(unique_HEs))
    HEs = unique_HEs[:num_HE]

    he_emb = np.zeros([max_num_he, word_emb_size], dtype=np.float32)
    he_emb_temp = [np.mean(word_vec[HE], axis=0) for HE in HEs]
    he_emb_temp = np.vstack(he_emb_temp)
    he_emb[:he_emb_temp.shape[0]] = he_emb_temp

    if mode=='train':
        return he_emb
    else:
        return he_emb, np.array(HEs)


def he_sampling_v2(adj, num_steps, max_num_he, eps_prob=0.001):
    """returns node indices of sampled hyperedge"""
    n_nodes = adj.shape[0]

    # binarize adj matrix
    adj = (adj > 0).astype('float')

    num_outedges = np.sum(adj, axis=1) + 0.5
    init_prob = num_outedges / np.sum(num_outedges, keepdims=True)

    adj = adj + eps_prob

    row_sum = np.sum(adj, axis=1, keepdims=True)
    adj = adj / row_sum

    start_node = np.random.choice(n_nodes, max_num_he*2, p=init_prob)

    HEs = [[n] for n in start_node]
    for k in range(num_steps - 1):
        [HE.append(sampling(adj[HE[-1]])[0]) for HE in HEs]

    unique_HEs = []
    for HE in HEs:
        unique_HEs.append(list(np.unique(HE)))
    if len(set([len(he) for he in unique_HEs])) == 1:  # np.unique automatically convert into array
        unique_HEs = np.unique(unique_HEs, axis=0)
    else:
        unique_HEs = np.unique(unique_HEs)

    num_HE = min(max_num_he, len(unique_HEs))
    HE_idx = np.random.choice(range(len(unique_HEs)), size=num_HE, replace=False)
    HEs = [unique_HEs[i] for i in HE_idx]
    return HEs


def get_word_vec(sg, vocab2idx, vocab_glove):
    node_labels = sg['node_labels']
    word_vec = []
    for node in node_labels:
        idx_glove = vocab2idx[node]
        word_vec.append( vocab_glove[idx_glove] )
    return np.vstack(word_vec)


def get_sim(vg_id_given, vg_id_compare,label, label_id2idx):
    idx_given = label_id2idx[vg_id_given]
    idx_compare = label_id2idx[vg_id_compare]
    sim = label[idx_given][idx_compare]

    return sim

def upsampling(sorted_vg_id, tail_range, num_sample_per_range):
    id_list = []

    pos_idx = np.random.randint(tail_range, size=num_sample_per_range)
    id_list.extend([sorted_vg_id[i].item() for i in pos_idx])

    neutral_idx = np.random.randint( len(sorted_vg_id)-2*tail_range, size=num_sample_per_range )
    id_list.extend([sorted_vg_id[i+tail_range].item() for i in neutral_idx])

    neg_idx = np.random.randint(tail_range, size=num_sample_per_range)
    id_list.extend([sorted_vg_id[-i].item() for i in neg_idx])

    return id_list


class DsetImgPairwise(Dataset):
    def __init__(self, dataset_name, sims, tail_range, split='train', transforms=None,
                 sample_mode='train'):
        if dataset_name == 'coco':
            self.ds = CocoDataset()
        elif dataset_name == 'f30k':
            self.ds = FlickrDataset()
        elif dataset_name == 'vg_coco':
            self.ds = VGDataset()
        else:
            raise ValueError(f'Invalid dataset_name {dataset_name}')

        self.sims = sims
        self.split = split
        self.l_id = self.ds.d_split[split]
        self.l_indices = [sims.id2idx[id_] for id_ in self.l_id]
        self.l_iter_id = None
        self.tail_range = tail_range  # if 0, do not use tail oversampling
        self.transforms = transforms
        self.sample_mode = sample_mode

    def __len__(self):
        if self.l_iter_id is None:
            return len(self.l_id)
        else:
            return len(self.l_iter_id)

    def __getitem__(self, idx):
        img_id = self.l_id[idx]
        anchor_img = self.get_by_id(img_id)
        pair_id, score = self.sample_pair(img_id)
        if self.sample_mode == 'tmb':
            pair_img = torch.stack([self.get_by_id(pid) for pid in pair_id])
            anchor_img = torch.cat([anchor_img.unsqueeze(0)] * len(pair_img))
            return anchor_img, pair_img, torch.tensor(score, dtype=torch.float)
        else:
            pair_img = self.get_by_id(pair_id)
            return anchor_img, pair_img, score

    def sample_pair(self, img_id):
        if self.sample_mode == 'train':
            scores = self.sims.sims[self.sims.id2idx[img_id]][self.l_indices]
            sorted_idx = np.argsort(scores)[::-1]  # decreasing order
            sorted_idx = sorted_idx[1:]  # exclude self
            if np.random.rand(1) < 0.5 and self.tail_range > 0:
                pair_idx = sorted_idx[np.random.randint(self.tail_range)]
            else:
                pair_idx = sorted_idx[np.random.randint(len(sorted_idx))]
            pair_id = self.l_id[pair_idx]
            score = scores[pair_idx]
            return pair_id, score
        elif self.sample_mode == 'three':
            scores = self.sims.sims[self.sims.id2idx[img_id]][self.l_indices]
            sorted_idx = np.argsort(scores)[::-1]  # decreasing order
            sorted_idx = sorted_idx[1:]  # exclude self
            r = np.random.rand(1)
            if r < 0.33 and self.tail_range > 0:
                pair_idx = sorted_idx[np.random.randint(self.tail_range)]
            elif 0.33 < r <= 0.66:
                pair_idx = sorted_idx[np.random.randint(self.tail_range,
                                                        high=len(sorted_idx) - self.tail_range)]
            else:
                pair_idx = sorted_idx[np.random.randint(len(sorted_idx) - self.tail_range,
                                                        high=len(sorted_idx))]
            pair_id = self.l_id[pair_idx]
            score = scores[pair_idx]
            return pair_id, score
        elif self.sample_mode == 'test':
            scores = self.sims.sims[self.sims.id2idx[img_id]][self.l_indices]
            sorted_idx = np.argsort(scores)[::-1]  # decreasing order
            sorted_idx = sorted_idx[1:]  # exclude self
            pair_idx = sorted_idx[np.random.randint(self.tail_range)]
            pair_id = self.l_id[pair_idx]
            score = scores[pair_idx]
            return pair_id, score
        elif self.sample_mode == 'tmb':
            scores = self.sims.sims[self.sims.id2idx[img_id]][self.l_indices]
            sorted_idx = np.argsort(scores)[::-1]  # decreasing order
            sorted_idx = sorted_idx[1:]  # exclude self
            N = len(sorted_idx)
            l_pair_id = []
            l_score = []
            # top
            for _ in range(2):
                pair_idx = sorted_idx[np.random.randint(self.tail_range)]
                pair_id = self.l_id[pair_idx]
                score = scores[pair_idx]
                l_pair_id.append(pair_id)
                l_score.append(score)
            # middle
            for _ in range(2):
                pair_idx = sorted_idx[np.random.randint(low=self.tail_range, high=N-self.tail_range)]
                pair_id = self.l_id[pair_idx]
                score = scores[pair_idx]
                l_pair_id.append(pair_id)
                l_score.append(score)
            # botten
            for _ in range(2):
                pair_idx = sorted_idx[np.random.randint(low=N-self.tail_range, high=N)]
                pair_id = self.l_id[pair_idx]
                score = scores[pair_idx]
                l_pair_id.append(pair_id)
                l_score.append(score)
            return l_pair_id, l_score

    def get_by_id(self, img_id):
        anchor_img = Image.open(self.ds.get_img_path(img_id)).convert('RGB')
        if self.transforms is not None:
            anchor_img = self.transforms(anchor_img)
        return anchor_img

    def set_iter_ids(self, l_img_id):  # depr
        """set list of ids to iterate"""
        self.l_iter_id = l_img_id


class DsetSGPairwise(Dataset):
    def __init__(self, ds, sims, tail_range, split='train', transforms=None,
                 mode='word', num_steps=3, max_num_he=100, sample_mode='tail_random',
                 bbox_size=64, n_rerank=200, pos_k=None):
        self.ds = ds
        assert isinstance(ds, CocoDataset) or isinstance(ds, FlickrDataset) or isinstance(ds, VGDataset),\
                'Requires dataset object'

        self.sims = sims
        self.split = split
        self.l_id = self.ds.d_split[split]
        self.l_indices = [sims.id2idx[id_] for id_ in self.l_id]
        self.l_iter_id = None
        self.tail_range = tail_range  # if 0, do not use tail oversampling
        self.transforms = transforms
        self.mode = mode
        self.sample_mode = sample_mode 
        self.num_steps = num_steps
        self.max_num_he = max_num_he
        self.bbox_size = bbox_size
        self.n_rerank = n_rerank
        self.pos_k = pos_k
        print(f'mode: {mode}, sample_mode: {sample_mode}, num_steps: {num_steps}, max_num_he: {max_num_he}, tail_range: {tail_range}, bbox_size: {bbox_size}')

        if self.mode in ('fixedbbox', 'bbox_rel', 'bbox_word'):
            normalize = TF.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            to_tensor = TF.ToTensor()
            resize = TF.Resize((bbox_size, bbox_size))
            self.tr = TF.Compose([resize, to_tensor, normalize])

    def __len__(self):
        if self.l_iter_id is None:
            return len(self.l_id)
        else:
            return len(self.l_iter_id)

    def __getitem__(self, idx):
        img_id = self.l_id[idx]
        anchor_data = self.get_by_id(img_id)
        if self.sample_mode == 'tmb':
            l_pair_id, score = self.sample_pair(img_id)
            l_pair_data = [self.get_by_id(pid) for pid in l_pair_id]
            pair_data = concat_data(l_pair_data)
            anchor_data = repeat_data(anchor_data, len(pair_data))
            return anchor_data, pair_data, torch.tensor(score, dtype=torch.float)
        elif self.sample_mode in ('tripletV1', 'rankV1'):
            pos_id, neg_id = self.sample_triplet(img_id)
            pos_data = self.get_by_id(pos_id)
            neg_data = self.get_by_id(neg_id)
            return anchor_data, pos_data, neg_data
        else:
            pair_id, score = self.sample_pair(img_id)
            pair_data = self.get_by_id(pair_id)
            return anchor_data, pair_data, score

    def sample_pair(self, img_id):
        if self.sample_mode == 'tail_random':
            scores = self.sims.sims[self.sims.id2idx[img_id]][self.l_indices]
            sorted_idx = np.argsort(scores)[::-1]  # decreasing order
            sorted_idx = sorted_idx[1:]  # exclude self
            if np.random.rand(1) < 0.5 and self.tail_range > 0:
                pair_idx = sorted_idx[np.random.randint(self.tail_range)]
            else:
                pair_idx = sorted_idx[np.random.randint(len(sorted_idx))]
            pair_id = self.l_id[pair_idx]
            score = scores[pair_idx]
            return pair_id, score
        elif self.sample_mode == 'tmb':
            scores = self.sims.sims[self.sims.id2idx[img_id]][self.l_indices]
            sorted_idx = np.argsort(scores)[::-1]  # decreasing order
            sorted_idx = sorted_idx[1:]  # exclude self
            N = len(sorted_idx)
            l_pair_id = []
            l_score = []
            # top
            for _ in range(2):
                pair_idx = sorted_idx[np.random.randint(self.tail_range)]
                pair_id = self.l_id[pair_idx]
                score = scores[pair_idx]
                l_pair_id.append(pair_id)
                l_score.append(score)
            # middle
            for _ in range(2):
                pair_idx = sorted_idx[np.random.randint(low=self.tail_range, high=N-self.tail_range)]
                pair_id = self.l_id[pair_idx]
                score = scores[pair_idx]
                l_pair_id.append(pair_id)
                l_score.append(score)
            # botten
            for _ in range(2):
                pair_idx = sorted_idx[np.random.randint(low=N-self.tail_range, high=N)]
                pair_id = self.l_id[pair_idx]
                score = scores[pair_idx]
                l_pair_id.append(pair_id)
                l_score.append(score)
            return l_pair_id, l_score
        elif self.sample_mode == 'rerank_random':
            """uniform-randomly select among resnet reranked images"""
            l_reranked = get_reranked_ids(self.ds.name, img_id, n_rerank=self.n_rerank, split=self.split)
            pair_id = l_reranked[np.random.randint(len(l_reranked))]
            score = self.sims.sims[self.sims.id2idx[img_id]][self.sims.id2idx[pair_id]]
            return pair_id, score
        else:
            raise ValueError(f'Invalid sample mode {self.sample_mode}')

    def sample_triplet(self, img_id):
        if self.sample_mode == 'tripletV1':
            l_reranked = get_reranked_ids(self.ds.name, img_id, n_rerank=self.n_rerank, split=self.split)
            # compute score
            l_score = [self.sims.get_similarity(img_id, id_) for id_ in l_reranked]
            # sort by score
            sort_idx = np.argsort(l_score)[::-1]
            # select top as pos
            pos_id = l_reranked[sort_idx[np.random.randint(self.pos_k)]]
            neg_id = l_reranked[sort_idx[np.random.randint(low=self.pos_k, high=len(l_reranked))]]
            # select bottom as neg
            return pos_id, neg_id
        elif self.sample_mode == 'rankV1':
            l_reranked = get_reranked_ids(self.ds.name, img_id, n_rerank=self.n_rerank, split=self.split)
            id1, id2 = np.random.choice(l_reranked, size=(2,), replace=False)
            sim1 = self.sims.get_similarity(img_id, id1)
            sim2 = self.sims.get_similarity(img_id, id2)
            if sim1 >= sim2:
                return id1, id2  # (pos_id, neg_id)
            else:
                return id2, id1  # (pos_id, neg_id)
        else:
            raise ValueError

    def get_by_id(self, img_id):
        if self.mode == 'adj':
            anchor_sg = self.ds.imgid2sg(img_id)
            X = torch.tensor([self.ds.word2emb(w) for w in anchor_sg['node_labels']], dtype=torch.float)
            adj = torch.tensor(anchor_sg['adj'], dtype=torch.float)
            # binarize and symmetrize
            symadj = ((adj + adj.T) > 0).to(torch.float)

            return {'adj': symadj,
                    'x': X,
                    'n_node': torch.tensor([len(X)])}
        elif self.mode == 'adj_he':
            anchor_sg = self.ds.imgid2sg(img_id)
            X = torch.tensor([self.ds.word2emb(w) for w in anchor_sg['node_labels']], dtype=torch.float)
            adj = torch.tensor(anchor_sg['adj'], dtype=torch.float)
            # binarize and symmetrize
            symadj = ((adj + adj.T) > 0).to(torch.float)

            he = he_sampling_v2(anchor_sg['adj'], self.num_steps, self.max_num_he)
            # pack hyperedge indices
            max_len = max([len(hh) for hh in he])
            he_tensor = - torch.ones(self.max_num_he, self.num_steps, dtype=torch.long)
            for i, row in enumerate(he):
                for j, idx in enumerate(row):
                    he_tensor[i, j] = torch.tensor(idx, dtype=torch.long)

            return {'adj': symadj,
                    'x': X,
                    'n_node': torch.tensor([len(X)]),
                    'he': he_tensor}
        else:
            # get scene graph
            anchor_sg = self.ds.imgid2sg(img_id)
            # sample hyperedge
            anchor_he = he_sampling_v2(anchor_sg['adj'], self.num_steps, self.max_num_he)
            # vocab 2 idx
            anchor_data = self.node2feature(anchor_he, img_id, anchor_sg)
            return anchor_data

    def node2feature(self, HEs, img_id, sg):
        if self.mode == 'fixedbbox':
            return self._bbox_feature(HEs, img_id, sg)
        elif self.mode == 'word':
            return self._word_feature(HEs, img_id, sg)
        elif self.mode == 'bbox_rel':
            return {'word': self._word_feature(HEs, img_id, sg, mode='rel'), 'bbox': self._bbox_feature(HEs, img_id, sg)}
        elif self.mode == 'bbox_word':
            return {'word': self._word_feature(HEs, img_id, sg, mode='word'), 'bbox': self._bbox_feature(HEs, img_id, sg)}
        else:
            raise ValueError(f'Invalid mode {self.mode}')

    def _bbox_feature(self, HEs, img_id, sg):
        img = Image.open(self.ds.get_img_path(img_id)).convert('RGB')
        if 'obj_bboxes' in sg:  # VG-COCO previously generated 
            bboxes = sg['obj_bboxes']
            r = 1  # for vg_coco dataset
        else:
            bboxes = sg['bboxes']
            w, h = img.size
            r = max(w, h) / 1024  # for bbox coordinate matching

        l_features = []
        for HE in HEs:
            l_row = [] 
            for node in HE:
                if node < len(bboxes):  # object
                    new_bb = bboxes[node] * r
                    crop_img = img.crop(box=new_bb)
                    bbox_tensor = self.tr(crop_img)
                    l_row.append(bbox_tensor)

            if len(l_row) == 0:  # no bbox found in this hyperedge
                continue
            row = torch.stack(l_row)
            row_box = -100 * torch.ones(self.num_steps, 3, self.bbox_size, self.bbox_size)  # at most num_steps - 1 bbox... but it isn't
            row_box[:len(row)] = row
            l_features.append(row_box)
        features = torch.stack(l_features) 
        out = - 100 * torch.ones(self.max_num_he, self.num_steps, 3, self.bbox_size, self.bbox_size)
        out[:len(features)] = features
        return out

    def _word_feature(self, HEs, img_id, sg, mode='word'):
        """mode: word -> all words, rel -> only relations"""
        l_features = []
        for HE in HEs:
            l_row = []
            for node in HE:
                if mode == 'word':
                    pass  # include all words
                elif mode == 'rel':
                    nodetype = self._node_type(node, sg)
                    if nodetype != 'rel':
                        continue
                else:
                    raise ValueError(f'Invalid mode {mode}')

                word = sg['node_labels'][node]
                l_row.append(torch.tensor(self.ds.word2emb(word), dtype=torch.float))

            if len(l_row) == 0:
                continue

            l_features.append(torch.mean(torch.stack(l_row), dim=0))
        features = torch.stack(l_features)
        out = torch.zeros(self.max_num_he, features.shape[1])
        out[:len(features), :] = features
        return out 

    def _node_type(self, node_id, sg):
        if 'obj_bboxes' in sg:  # VG-COCO previously generated
            if node_id < len(sg['obj_bboxes']):
                return 'obj'
            elif len(sg['obj_bboxes']) <= node_id < len(sg['obj_attributes']):
                return 'attr'
            else:
                return 'rel'
        else:
            if node_id < len(sg['bboxes']):
                return 'obj'
            else:
                return 'rel'


def concat_data(l_x):
    if isinstance(l_x[0], torch.Tensor):
        return torch.stack(l_x)
    elif isinstance(l_x[0], dict):
        out = {}
        for key in l_x[0].keys():
            l_data = [x[key] for x in l_x]
            if len(set([d.shape for d in l_data])) == 1:
                out[key] = torch.stack(l_data)
            else:
                # assume tensors are 2-dimensional : Adj mats, Feature mats
                D1 = max([x.shape[0] for x in l_data])
                D2 = max([x.shape[1] for x in l_data])
                A = torch.zeros(len(l_data), D1, D2)
                for i, x in enumerate(l_data):
                    A[i, :x.shape[0], :x.shape[1]] = x
                out[key] = A
        # node flat
        out['nodes_flat'] = torch.cat([x['x'] for x in l_x])

        # adj flat & batch
        l_adj = [x['adj'] for x in l_x]
        n_total_nodes = sum([len(adj) for adj in l_adj])
        adj_flat = torch.zeros(n_total_nodes, n_total_nodes)
        start = 0
        for adj in l_adj:
            end = len(adj)
            adj_flat[start:start + end, start:start + end]  = adj
            start += end
        out['adj_flat'] = adj_flat
        out['batch'] = torch.cat([torch.ones(len(adj)) * i for i, adj in enumerate(l_adj)]).long()
        return out
    elif isinstance(l_x[0], tuple):
        return [ concat_data([x[i] for x in l_x]) for i in range(len(l_x[0]))]
    else:
        return torch.tensor(l_x)



def repeat_data(x, n_repeat):
    if isinstance(x, torch.Tensor):
        return torch.stack([x] * n_repeat)
    elif isinstance(x, dict):
        return concat_data([x for i in range(n_repeat)])
        #out = {}
        # for key in x.keys():
        #     out[key] = torch.stack([x[key]] * n_repeat)
        # return out


class Dset_VG_Pairwise(Dataset):
    def __init__(self, cfg, sg, label, label_ids, vocab_glove, vocab2idx, mode):
        self.cfg = cfg
        self.max_num_he = cfg['MODEL']['NUM_MAX_HE']
        self.word_emb_size = cfg['MODEL']['WORD_EMB_SIZE']
        self.sampling_steps = cfg['MODEL']['STEP']

        self.sg = sg
        self.sg_keys = list(self.sg.keys())
        self.label = label
        self.vg_id_list = label_ids
        self.label_id2idx = {str(val): i for i, val in enumerate(label_ids)}

        self.label_id2idx_split = [self.label_id2idx[id] for id in self.sg_keys]
        self.tail_range = cfg['MODEL']['TAIL_RANGE']
        self.num_sample_per_range = cfg['MODEL']['NUM_SAMPLE_PER_RANGE']
        self.vocab_glove = vocab_glove
        self.vocab2idx = vocab2idx
        self.mode = mode

    def __len__(self):
        return len(self.sg)

    def __getitem__(self, idx):
        vg_img_id = self.sg_keys[idx]
        sg_anchor = self.sg[vg_img_id]

        score = self.label[self.label_id2idx[vg_img_id]][self.label_id2idx_split]
        sorted_idx = np.argsort(score)[::-1]

        compare_img_idx = upsampling(sorted_idx, self.tail_range, self.num_sample_per_range)
        #compare_img_idx = [pos_vg_idx.item(), neutral_vg_idx.item(), neg_vg_idx.item()]
        #compare_img = np.random.randint(len(self.sg), size=1)

        compare_img_id = [ self.sg_keys[i] for i in compare_img_idx]
        sim_score = [ get_sim(vg_img_id, i, self.label, self.label_id2idx) for i in compare_img_id]
        sg_compare = np.array([ self.sg[i] for i in compare_img_id ])

        word_vec_anchor = get_word_vec(sg_anchor, self.vocab2idx, self.vocab_glove)
        word_vec_compare = [ get_word_vec(i, self.vocab2idx, self.vocab_glove) for i in sg_compare]

        HE_anchor = np.array(
                    [he_sampling(sg_anchor['adj'], word_vec_anchor, self.sampling_steps, self.max_num_he, 'train')
                                   for i in range(len(compare_img_idx)) ]
                    )
        HE_compare = np.array(
                    [he_sampling(sg_compare[i]['adj'], word_vec_compare[i],self.sampling_steps, self.max_num_he, 'train')
                                    for i in range(len(compare_img_idx)) ]
                    )

        return HE_anchor, HE_compare, np.reshape(sim_score, [-1, 1])


class Dset_VG_Pairwise_AUX(Dataset):
    def __init__(self, cfg, sg, label, label_ids, vocab_glove, vocab2idx, mode):
        self.cfg = cfg
        self.max_num_he = cfg['MODEL']['NUM_MAX_HE']
        self.word_emb_size = cfg['MODEL']['WORD_EMB_SIZE']
        self.sampling_steps = cfg['MODEL']['STEP']

        self.sg = sg
        self.sg_keys = list(self.sg.keys())
        self.label = label
        self.vg_id_list = label_ids
        self.label_id2idx = {str(val): i for i, val in enumerate(label_ids)}

        self.label_id2idx_split = [self.label_id2idx[id] for id in self.sg_keys]
        self.tail_range = cfg['MODEL']['TAIL_RANGE']
        self.num_sample_per_range = cfg['MODEL']['NUM_SAMPLE_PER_RANGE']
        self.vocab_glove = vocab_glove
        self.vocab2idx = vocab2idx
        self.mode = mode

        self.resnet_dir = '/data/public/rw/datasets/visual_genome/Resnet_feature/wholeImg/'

    def __len__(self):
        return len(self.sg)

    def __getitem__(self, idx):
        vg_img_id = self.sg_keys[idx]
        sg_anchor = self.sg[vg_img_id]

        score = self.label[self.label_id2idx[vg_img_id]][self.label_id2idx_split]
        sorted_idx = np.argsort(score)[::-1]

        compare_img_idx = upsampling(sorted_idx, self.tail_range, self.num_sample_per_range)
        #compare_img_idx = [pos_vg_idx.item(), neutral_vg_idx.item(), neg_vg_idx.item()]
        #compare_img = np.random.randint(len(self.sg), size=1)

        compare_img_id = [ self.sg_keys[i] for i in compare_img_idx]
        sim_score = [ get_sim(vg_img_id, i, self.label, self.label_id2idx) for i in compare_img_id]
        sg_compare = np.array([ self.sg[i] for i in compare_img_id ])

        word_vec_anchor = get_word_vec(sg_anchor, self.vocab2idx, self.vocab_glove)
        word_vec_compare = [ get_word_vec(i, self.vocab2idx, self.vocab_glove) for i in sg_compare]

        aux_anchor = self.get_resnet_feature([vg_img_id])
        aux_anchor = np.tile(aux_anchor, (len(compare_img_id), 1))
        aux_compare = self.get_resnet_feature(compare_img_id)

        HE_anchor = np.array(
                    [he_sampling(sg_anchor['adj'], word_vec_anchor, self.sampling_steps, self.max_num_he, 'train')
                                   for i in range(len(compare_img_idx)) ]
                    )
        HE_compare = np.array(
                    [he_sampling(sg_compare[i]['adj'], word_vec_compare[i],self.sampling_steps, self.max_num_he, 'train')
                                    for i in range(len(compare_img_idx)) ]
                    )

        return HE_anchor, HE_compare, np.reshape(sim_score, [-1, 1]), aux_anchor, aux_compare

    def get_resnet_feature(self, l_img_id):
        return np.array([np.load(os.path.join(self.resnet_dir, f'{img_id}.npy')) for img_id in l_img_id])



class Dset_VG_inference(Dataset):
    def __init__(self, cfg, sg, label, label_ids, vocab_glove, vocab2idx):
        self.cfg = cfg
        self.max_num_he = cfg['MODEL']['NUM_MAX_HE']
        self.word_emb_size = cfg['MODEL']['WORD_EMB_SIZE']
        self.sampling_steps = cfg['MODEL']['STEP']

        self.sg = sg
        self.sg_keys = list(self.sg.keys())
        self.label = label
        self.label_id2idx = {str(val): i for i, val in enumerate(label_ids)}

        self.vocab_glove = vocab_glove
        self.vocab2idx = vocab2idx

    def __len__(self):
        return len(self.sg)

    def __getitem__(self, idx):
        vg_img_id = self.sg_keys[idx]
        sg_compare = self.sg[vg_img_id]

        word_vec_compare = get_word_vec(sg_compare, self.vocab2idx, self.vocab_glove)

        HE_compare, HEs = he_sampling(sg_compare['adj'], word_vec_compare, self.sampling_steps, self.max_num_he, 'infer')

        return HE_compare, vg_img_id


class Dset_VG_inference_AUX(Dataset):
    def __init__(self, cfg, sg, label, label_ids, vocab_glove, vocab2idx):
        self.cfg = cfg
        self.max_num_he = cfg['MODEL']['NUM_MAX_HE']
        self.word_emb_size = cfg['MODEL']['WORD_EMB_SIZE']
        self.sampling_steps = cfg['MODEL']['STEP']

        self.sg = sg
        self.sg_keys = list(self.sg.keys())
        self.label = label
        self.label_id2idx = {str(val): i for i, val in enumerate(label_ids)}

        self.vocab_glove = vocab_glove
        self.vocab2idx = vocab2idx

        self.resnet_dir = '/data/public/rw/datasets/visual_genome/Resnet_feature/wholeImg/'

    def __len__(self):
        return len(self.sg)

    def __getitem__(self, idx):
        vg_img_id = self.sg_keys[idx]
        sg_compare = self.sg[vg_img_id]

        word_vec_compare = get_word_vec(sg_compare, self.vocab2idx, self.vocab_glove)

        HE_compare, HEs = he_sampling(sg_compare['adj'], word_vec_compare, self.sampling_steps, self.max_num_he, 'infer')
        aux_compare = self.get_resnet_feature([vg_img_id])[0]

        return HE_compare, vg_img_id, aux_compare

    def get_resnet_feature(self, l_img_id):
        return np.array([np.load(os.path.join(self.resnet_dir, f'{img_id}.npy')) for img_id in l_img_id])

#for triplet loss
class Dset_VG(Dataset):
    def __init__(self, cfg, sg, label, label_ids, vocab_glove, vocab2idx):
        self.cfg =cfg
        self.max_num_he = cfg['MODEL']['NUM_MAX_HE']
        self.word_emb_size = cfg['MODEL']['WORD_EMB_SIZE']
        self.sampling_steps = cfg['MODEL']['STEP']

        self.sg = sg
        self.sg_keys = list(self.sg.keys())
        self.label = label
        self.label_id2idx = {str(val): i for i, val in enumerate(label_ids)}

        self.vocab_glove = vocab_glove
        self.vocab2idx = vocab2idx

    def __len__(self):
        return len(self.sg)

    def __getitem__(self, idx):
        vg_img_id = self.sg_keys[idx]
        sg_anchor = self.sg[vg_img_id]
        compare_imgs = np.random.randint(len(self.sg), size=2)
        compare_img_ids = [self.sg_keys[i] for i in compare_imgs]
        sim_score = [get_sim(vg_img_id, i, self.label, self.label_id2idx) for i in compare_img_ids]

        if sim_score[0] < sim_score[1]:
            sg_pos = self.sg[compare_img_ids[1]]
            sg_neg = self.sg[compare_img_ids[0]]
        else:
            sg_pos = self.sg[compare_img_ids[0]]
            sg_neg = self.sg[compare_img_ids[1]]

        word_vec_anchor = get_word_vec(sg_anchor, self.vocab2idx, self.vocab_glove)
        word_vec_pos = get_word_vec(sg_pos, self.vocab2idx, self.vocab_glove)
        word_vec_neg = get_word_vec(sg_neg, self.vocab2idx, self.vocab_glove)

        HE_anchor = he_sampling(sg_anchor['adj'], word_vec_anchor, self.sampling_steps, self.max_num_he)
        HE_pos = he_sampling(sg_pos['adj'], word_vec_pos, self.sampling_steps, self.max_num_he)
        HE_neg = he_sampling(sg_neg['adj'], word_vec_neg, self.sampling_steps, self.max_num_he)

        return HE_anchor, HE_pos, HE_neg







