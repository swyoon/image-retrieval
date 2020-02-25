import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
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

    def forward(self, he_anchor, he_pos, he_neg, mode='train', **kwargs):
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


class HGAN_AUX(HGAN):
    """use an auxiliary feature vector (primarily resnet feature) as an additional hyperedge"""
    def __init__(self, cfg):
        super(HGAN_AUX, self).__init__(cfg)

        self.n_aux = cfg['MODEL']['NUM_AUX']
        # self.aux2h = torch.nn.Linear(self.n_aux, self.nhidden)
        self.aux2emb = torch.nn.Linear(self.n_aux, self.word_emb_size)

        self.onlyres = cfg['MODEL'].get('ONLYRES', False)  # using only resnet feature
        if self.onlyres:
            print('using only resnet features')

    def score(self, he_anchor, he_compare, aux_anchor, aux_compare):
        aux_emb_anchor = self.aux2emb(aux_anchor).unsqueeze(1)
        aux_emb_compare = self.aux2emb(aux_compare).unsqueeze(1)
        if self.onlyres:
            cat_anchor = torch.cat([aux_emb_anchor], dim=1)
            cat_compare = torch.cat([aux_emb_compare], dim=1)
        else:
            cat_anchor = torch.cat([he_anchor, aux_emb_anchor], dim=1)
            cat_compare = torch.cat([he_compare, aux_emb_compare], dim=1)
        return super(HGAN_AUX, self).score(cat_anchor, cat_compare)

    def forward(self, he_anchor, he_pos, he_neg, aux_anchor=None, aux_pos=None, aux_neg=None, mode='train'):

        if self.cfg['MODEL']['TARGET'] == 'SBERT':
            # he_neg is SBERT score
            score_bert = torch.reshape(he_neg, (-1, 1))
            score_p, att_map = self.score(he_anchor, he_pos, aux_anchor, aux_pos)
            loss = self.loss(score_p, score_bert)
            if mode=='train':
                return score_p, loss
            else:
                return score_p, loss, att_map

        else:
            raise NotImplementedError


class HGAN_LATE_V1(HGAN):
    """late fusion between auxiliary feature
    v1: HGAN predicts the difference between cosine similarity"""
    def __init__(self, cfg):
        super(HGAN_LATE_V1, self).__init__(cfg)

    def score(self, he_anchor, he_compare, aux_anchor, aux_compare):
        aux_anchor_norm = torch.norm(aux_anchor, dim=1)
        aux_compare_norm = torch.norm(aux_compare, dim=1)
        aux_cosine_sim = (aux_anchor * aux_compare).sum(dim=1)  / aux_anchor_norm / aux_compare_norm
        aux_cosine_sim += 1  # make it positive, following S-BERT score
        aux_cosine_sim = aux_cosine_sim.unsqueeze(1)

        he_score, he_att_map = super(HGAN_LATE_V1, self).score(he_anchor, he_compare)
        return aux_cosine_sim + he_score, he_att_map

    def forward(self, he_anchor, he_pos, he_neg, aux_anchor=None, aux_pos=None, aux_neg=None, mode='train'):

        if self.cfg['MODEL']['TARGET'] == 'SBERT':
            # he_neg is SBERT score
            score_bert = torch.reshape(he_neg, (-1, 1))
            score_p, att_map = self.score(he_anchor, he_pos, aux_anchor, aux_pos)
            loss = self.loss(score_p, score_bert)
            if mode=='train':
                return score_p, loss
            else:
                return score_p, loss, att_map

        else:
            raise NotImplementedError


class HGAN_V2(HGAN):
    """late fusion between auxiliary feature
    v1: HGAN predicts the difference between cosine similarity"""
    def __init__(self, cfg):
        super(HGAN_V2, self).__init__(cfg)
        self.resnetweight = cfg['MODEL']['RESNET_WEIGHT']

    def score(self, he_anchor, he_compare, aux_anchor, aux_compare):
        aux_anchor_norm = torch.norm(aux_anchor, dim=1)
        aux_compare_norm = torch.norm(aux_compare, dim=1)
        aux_cosine_sim = (aux_anchor * aux_compare).sum(dim=1)  / aux_anchor_norm / aux_compare_norm
        aux_cosine_sim += 1  # make it positive, following S-BERT score
        aux_cosine_sim = aux_cosine_sim.unsqueeze(1)

        he_score, he_att_map = super(HGAN_LATE_V1, self).score(he_anchor, he_compare)
        return aux_cosine_sim * self.resnetweight + he_score, he_att_map

    def forward(self, he_anchor, he_pos, he_neg, aux_anchor=None, aux_pos=None, aux_neg=None, mode='train'):

        if self.cfg['MODEL']['TARGET'] == 'SBERT':
            # he_neg is SBERT score
            score_bert = torch.reshape(he_neg, (-1, 1))
            score_p, att_map = self.score(he_anchor, he_pos, aux_anchor, aux_pos)
            loss = self.loss(score_p, score_bert)
            if mode=='train':
                return score_p, loss
            else:
                return score_p, loss, att_map

        else:
            raise NotImplementedError



class DeepMetric(nn.Module):
    def __init__(self, backbone='resnet152', finetune=False, normalize=False,
                 mode='regression', margin=None, embedding_dim=1024):
        super().__init__()
        self.backbone = backbone
        self.finetune = finetune
        self.normalize = normalize
        self.mode = mode
        self.margin = margin
        self.embed = nn.Conv2d(2048, embedding_dim, 1)
        assert mode in ('regression', 'triplet')
        if backbone == 'resnet152':
            resnet = torchvision.models.resnet152(pretrained=True)
            feature_part = list(resnet.children())[:-1]
            net = nn.Sequential(*feature_part)

        if not finetune:
            for p in net.parameters():
                p.requires_grad = False
        self.net = net

    def forward(self, img1, img2):
        rep1 = self.net(img1)
        rep1 = self.embed(rep1)
        rep2 = self.net(img2)
        rep2 = self.embed(rep2)
        if self.normalize:
            rep1 = rep1 / rep1.norm(dim=1, keepdim=True)
            rep2 = rep2 / rep2.norm(dim=1, keepdim=True)
        sim = (rep1 * rep2).sum(dim=1)
        return sim

    def loss(self, img1, img2, score_or_img3):
        if self.mode == 'regression':
            score = score_or_img3
            sim = self(img1, img2)
            return ((score - sim) ** 2).mean()
        elif self.mode == 'triplet':
            img3 = score_or_img3
            sim1 = self(img1, img2)  # pos
            sim2 = self(img1, img3)  # neg
            return F.relu(- sim1 + sim2 + self.margin).mean()



