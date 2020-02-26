import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as TF
from torch.utils.data import Dataset
import operator
import os
from data import CocoDataset, FlickrDataset, VGDataset
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
                 mode='word', num_steps=3, max_num_he=100, sample_mode='tail_random'):
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
        print(f'mode: {mode}, sample_mode: {sample_mode}, num_steps: {num_steps}, max_num_he: {max_num_he}, tail_range: {tail_range}')

        self.tr_normalize = TF.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.tr_to_tensor = TF.ToTensor()
        if self.mode == 'fixedbbox':
            resnet = models.resnet18(pretrained=True)
            feature_part = list(resnet.children())[:-1]
            resnet = nn.Sequential(*feature_part)
            resnet = resnet
            resnet.eval()
            self.resnet = resnet

    def __len__(self):
        if self.l_iter_id is None:
            return len(self.l_id)
        else:
            return len(self.l_iter_id)

    def __getitem__(self, idx):
        img_id = self.l_id[idx]
        anchor_data = self.get_by_id(img_id)
        pair_id, score = self.sample_pair(img_id)
        if self.sample_mode == 'tmb':
            pair_data = torch.stack([self.get_by_id(pid) for pid in pair_id])
            anchor_data = anchor_data.unsqueeze(0).repeat(len(pair_data), 1, 1)
            return anchor_data, pair_data, torch.tensor(score, dtype=torch.float)
        else:
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
        else:
            raise ValueError(f'Invalid sample mode {self.sample_mode}')

    def get_by_id(self, img_id):
        # get scene graph
        anchor_sg = self.ds.imgid2sg(img_id)
        # sample hyperedge
        anchor_he = he_sampling_v2(anchor_sg['adj'], self.num_steps, self.max_num_he)
        # vocab 2 idx
        anchor_data = self.node2feature(anchor_he, img_id, anchor_sg)
        return anchor_data

    def node2feature(self, HEs, img_id, sg):
        if self.mode == 'fixedbbox':
            img = Image.open(self.ds.get_img_path(img_id)).convert('RGB')
            # w, h = img.size
            # r = max(w, h) / 1024  # for bbox coordinate matching
            r = 1  # for vg_coco dataset

            l_features = []
            for HE in HEs:
                l_row = [] 
                for node in HE:
                    if node < len(sg['obj_bboxes']):  # object
                        new_bb = sg['obj_bboxes'][node] * r
                        cropped = img.crop(box=new_bb)
                        img_tensor = self.tr_normalize(self.tr_to_tensor(cropped))
                        img_tensor = img_tensor.unsqueeze(0)# .cuda()
                        feat = self.resnet(img_tensor)
                        feat = feat.squeeze(3).squeeze(2)[0]
                        l_row.append(feat)
                if len(l_row) == 0:  # no bbox found in this hyperedge
                    continue
                l_features.append(torch.mean(torch.stack(l_row), dim=0))
            features = torch.stack(l_features)
            out = torch.zeros(self.max_num_he, features.shape[1])
            out[:len(features), :] = features
            return out 
        elif self.mode == 'word':
            l_features = []
            for HE in HEs:
                l_row = []
                for node in HE:
                    word = sg['node_labels'][node]
                    l_row.append(torch.tensor(self.ds.word2emb(word), dtype=torch.float))
                l_features.append(torch.mean(torch.stack(l_row), dim=0))
            features = torch.stack(l_features)
            out = torch.zeros(self.max_num_he, features.shape[1])
            out[:len(features),:] = features
            return out 


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







