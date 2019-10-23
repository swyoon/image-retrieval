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

class Dset_VG(Dataset):
    def __init__(self, cfg, sg, label, label_ids, vocab_glove, vocab2idx, mode):
        self.max_num_he = cfg['MODEL']['NUM_MAX_HE']
        self.word_emb_size = cfg['MODEL']['WORD_EMB_SIZE']
        self.sampling_steps = cfg['MODEL']['STEP']

        self.sg = sg
        self.sg_keys = list(self.sg.keys())
        self.label = label
        self.label_id2idx = {str(val): i for i, val in enumerate(label_ids)}
        self.max_len_q = 20

        self.vocab_glove = vocab_glove
        self.vocab2idx = vocab2idx
        self.mode = mode

    def __len__(self):
        return len(self.sg)

    def __getitem__(self, idx):
        vg_img_id = self.sg_keys[idx]
        sg_anchor = self.sg[vg_img_id]
        compare_imgs = np.random.randint(len(self.sg), size=2)
        compare_img_ids = [self.sg_keys[i] for i in compare_imgs]
        sim_score = [self.get_sim(vg_img_id, i) for i in compare_img_ids]

        if sim_score[0] < sim_score[1]:
            sg_pos = self.sg[compare_img_ids[1]]
            sg_neg = self.sg[compare_img_ids[0]]
        else:
            sg_pos = self.sg[compare_img_ids[0]]
            sg_neg = self.sg[compare_img_ids[1]]

        word_vec_anchor = self.get_word_vec(sg_anchor)
        word_vec_pos = self.get_word_vec(sg_pos)
        word_vec_neg = self.get_word_vec(sg_neg)

        HE_anchor = he_sampling(sg_anchor['adj'], word_vec_anchor, self.sampling_steps, self.max_num_he)
        HE_pos = he_sampling(sg_pos['adj'], word_vec_pos, self.sampling_steps, self.max_num_he)
        HE_neg = he_sampling(sg_neg['adj'], word_vec_neg, self.sampling_steps, self.max_num_he)

        return HE_anchor, HE_pos, HE_neg

    def get_word_vec(self, sg):
        node_labels = sg['node_labels']
        word_vec = []
        for node in node_labels:
            idx_glove = self.vocab2idx[node]
            word_vec.append( self.vocab_glove[idx_glove] )
        return np.vstack(word_vec)

    def get_sim(self, given, compare):
        idx_given = self.label_id2idx[given]
        idx_compare = self.label_id2idx[compare]
        sim = self.label[idx_given][idx_compare]

        return sim





