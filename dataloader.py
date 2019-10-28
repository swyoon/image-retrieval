import numpy as np
from torch.utils.data import Dataset


def sampling(prob):
    return np.random.choice(len(prob), 1, p=prob)


def he_sampling(adj, word_vec, num_steps, max_num_he, eps_prob=0.001, word_emb_size=300):
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

    return he_emb

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


class Dset_VG_Pairwise(Dataset):
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
        sg_anchor = self.sg[vg_img_id]

        compare_img = np.random.randint(len(self.sg), size=1)
        compare_img_id = self.sg_keys[compare_img[0]]
        sim_score = get_sim(vg_img_id, compare_img_id, self.label, self.label_id2idx)
        sg_compare = self.sg[compare_img_id]

        word_vec_anchor = get_word_vec(sg_anchor, self.vocab2idx, self.vocab_glove)
        word_vec_compare = get_word_vec(sg_compare, self.vocab2idx, self.vocab_glove)

        HE_anchor = he_sampling(sg_anchor['adj'], word_vec_anchor, self.sampling_steps, self.max_num_he)
        HE_compare = he_sampling(sg_compare['adj'], word_vec_compare, self.sampling_steps, self.max_num_he)

        return HE_anchor, HE_compare, sim_score


class Dset_VG_inference(Dataset):
    def __init__(self, cfg, train_sg, label, label_ids, vocab_glove, vocab2idx):
        self.cfg = cfg
        self.max_num_he = cfg['MODEL']['NUM_MAX_HE']
        self.word_emb_size = cfg['MODEL']['WORD_EMB_SIZE']
        self.sampling_steps = cfg['MODEL']['STEP']

        self.train_sg = train_sg
        self.sg_keys = list(self.train_sg.keys())
        self.label = label
        self.label_id2idx = {str(val): i for i, val in enumerate(label_ids)}

        self.vocab_glove = vocab_glove
        self.vocab2idx = vocab2idx

    def __len__(self):
        return len(self.train_sg)

    def __getitem__(self, idx):
        vg_img_id = self.sg_keys[idx]
        sg_compare = self.train_sg[vg_img_id]

        word_vec_compare = get_word_vec(sg_compare, self.vocab2idx, self.vocab_glove)

        HE_compare = he_sampling(sg_compare['adj'], word_vec_compare, self.sampling_steps, self.max_num_he)

        return HE_compare, vg_img_id


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







