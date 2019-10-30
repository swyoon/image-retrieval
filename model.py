import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, score_pos, score_neg, size_average=True):
        losses = F.relu(score_neg - score_pos + self.margin)
        return losses.mean() if size_average else losses.sum()

class HGAN(nn.Module):
    def __init__(self, cfg):
        super(HGAN, self).__init__()

        self.cfg = cfg
        self.word_emb_size = cfg['MODEL']['WORD_EMB_SIZE']
        self.nhidden = cfg['MODEL']['NUM_HIDDEN']
        self.nhead = cfg['MODEL']['NUM_HEAD']
        self.a2h = torch.nn.Linear(self.word_emb_size, self.nhidden)
        self.c2h = torch.nn.Linear(self.word_emb_size, self.nhidden)

        self.h2att = torch.nn.Linear(self.nhidden, self.nhead)
        self.softmax_att = torch.nn.Softmax(dim=2)
        self.fc_score = torch.nn.Linear(self.nhidden*self.nhead, 1)

        #self.margin = cfg['MODEL']['LOSS_MARGIN']
        if self.cfg['MODEL']['TARGET'] == 'SBERT':
            self.loss = nn.MSELoss()
        else:
            self.loss = TripletLoss(self.margin)

    def score(self, he_anchor, he_compare):
        num_he_anchor = he_anchor.shape[1]
        num_he_compare = he_compare.shape[1]

        he_anchor = self.a2h(he_anchor)
        he_compare = self.c2h(he_compare)

        he_anchor = he_anchor.permute(0, 2, 1)  # [B, d, #a]
        he_compare = he_compare.permute(0, 2, 1)  # [B, d, #p]

        he_anchor_selfatt = he_anchor.unsqueeze(3)  # [B, d, #a, 1]
        he_compare_selfatt = he_compare.unsqueeze(2)  # [B, d, 1, #p]

        self_mul = torch.matmul(he_anchor_selfatt, he_compare_selfatt)  # [B, d, #a, #p]
        self_mul = self_mul.permute(0, 2, 3, 1)  # [B, #a, #p, d]

        att_map = self.h2att(self_mul)  # [B, #a, #p, h]
        att_map = att_map.permute(0, 3, 1, 2)  # [B, h, #a, #p]
        att_map = torch.reshape(att_map, (-1, self.nhead, num_he_anchor * num_he_compare))  # [B, h, #a*#p]
        att_map = self.softmax_att(att_map)
        att_map = torch.reshape(att_map, (-1, self.nhead, num_he_anchor, num_he_compare))  # [B, h, #a, #p]

        he_anchor = he_anchor.unsqueeze(2)  # [B, d, 1, #a]
        he_compare = he_compare.unsqueeze(3)  # [B, d, #p, 1]

        for i in range(self.nhead):
            att_g = att_map[:, i:i + 1, :, :]  # [B, 1, #a, #p]
            att_g_t = att_g.repeat([1, self.nhidden, 1, 1])  # [B, d, #a, #p]
            att_out = torch.matmul(he_anchor, att_g_t)  # [B, d, 1, #p]
            att_out = torch.matmul(att_out, he_compare)  # [B, d, 1, 1]
            att_out = att_out.squeeze(-1)
            att_out_sq = att_out.squeeze(-1)  # [B, d]

            if i == 0:
                out = att_out_sq
            else:
                out = torch.cat((out, att_out_sq), dim=1)  # [B, d*h]

        score = self.fc_score(out)
        #score = F.relu(score)
        return score, att_map

    def forward(self, he_anchor, he_pos, he_neg, mode='train'):
        #he_*: B x 100 x dim_word_emb

        if self.cfg['MODEL']['TARGET'] == 'SBERT':
            score_bert = torch.reshape(he_neg, (-1, 1))
            score_p, att_map = self.score(he_anchor, he_pos)
            loss = self.loss(score_p, score_bert)
            if mode=='train':
                return score_p, loss
            else:
                return score_p, loss, att_map
        else:
            score_p,_ = self.score(he_anchor, he_pos)
            score_n,_ = self.score(he_anchor, he_neg)

            loss = self.loss(score_p, score_n)

            return score_p, score_n, loss
