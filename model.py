import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from torchvision.models import resnet18
import torch_geometric.nn as gnn

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
        self.ver = cfg['MODEL'].get('VER', 1)
        self.out_activation = cfg['MODEL'].get('OUTACT', 'linear')
        self.word_emb_size = cfg['MODEL']['WORD_EMB_SIZE']
        self.nhidden = cfg['MODEL']['NUM_HIDDEN']
        self.nhead = cfg['MODEL']['NUM_HEAD']
        self.a2h = torch.nn.Linear(self.word_emb_size, self.nhidden)
        if self.ver == 1:
            self.c2h = torch.nn.Linear(self.word_emb_size, self.nhidden)

        self.h2att = torch.nn.Linear(self.nhidden, self.nhead)
        self.softmax_att = torch.nn.Softmax(dim=2)
        self.fc_score = torch.nn.Linear(self.nhidden*self.nhead, 1)

        resnet_feature_dim = 512
        glove_feature_dim = 300

        if self.cfg['MODEL']['TARGET'] == 'SBERT':
            self.loss = nn.MSELoss()
        else:
            self.margin = cfg['MODEL']['LOSS_MARGIN']
            self.loss = TripletLoss(self.margin)

        if 'FEATURE' in self.cfg['MODEL']:
            if self.cfg['MODEL']['FEATURE'] in ('bbox', 'bbox_rel', 'bbox_word', 'bbox_rel_cat', 'bbox_word_cat'):
                resnet = resnet18(pretrained=True)
                feature_part = list(resnet.children())[:-1]
                net = nn.Sequential(*feature_part)
                self.cnn = net

                if not self.cfg['MODEL'].get('FINETUNE', True):
                    for p in self.cnn.parameters():
                        p.requires_grad = False 

            if self.cfg['MODEL']['FEATURE'] in ('bbox_rel', 'bbox_word', 'bbox_rel_cat', 'bbox_word_cat'):
                self.embed_bbox = torch.nn.Linear(resnet_feature_dim, self.word_emb_size)
                self.embed_word = torch.nn.Linear(glove_feature_dim, self.word_emb_size)

    def _preprocess(self, x):
        if 'FEATURE' not in self.cfg['MODEL'] or self.cfg['MODEL']['FEATURE'] == 'word':
            return self._word_feature(x)
        if self.cfg['MODEL']['FEATURE'] == 'bbox':
            return self._bbox_feature(x)
        elif self.cfg['MODEL']['FEATURE'] in ('bbox_rel', 'bbox_word'):  # bbox with relation word
            bbox_feature = self._bbox_feature(x['bbox'])
            word_feature = self._word_feature(x['word'])
            return self.embed_bbox(bbox_feature) + self.embed_word(word_feature)
        elif self.cfg['MODEL']['FEATURE'] in ('bbox_rel_cat', 'bbox_word_cat'):  # bbox with relation word
            bbox_feature = self._bbox_feature(x['bbox'])
            word_feature = self._word_feature(x['word'])
            return torch.cat([self.embed_bbox(bbox_feature), self.embed_word(word_feature)], dim=1)
        else:
            raise ValueError

    def _word_feature(self, x):
        if len(x.shape) == 4:
            return x.view(-1, *x.shape[2:])
        else:
            return x

    def _bbox_feature(self, x):
        """x: 6D tensor of bbox images
        returns averages of resnet vectors"""
        in_shape = x.shape
        if len(in_shape) == 6:
            x = x.view(-1, *in_shape[3:])
        mask = x[:, 0, 0, 0] != -100
        real_x = x[mask]
        feature = self.cnn(real_x).squeeze(2).squeeze(1)
        out = torch.zeros(in_shape[0] * in_shape[1] * in_shape[2], feature.shape[1]).to(x.device)
        out = out.masked_scatter_(mask[:, None], feature)
        out = out.view(in_shape[0], in_shape[1], in_shape[2], feature.shape[1])
        num = mask.view(*in_shape[:3]).sum(dim=2).unsqueeze(-1).to(torch.float)
        num.clamp_(min=1)
        out = out.sum(dim=2) / num
        return out

    def score(self, he_anchor, he_compare):
        he_anchor = self._preprocess(he_anchor)
        he_compare = self._preprocess(he_compare)
        num_he_anchor = he_anchor.shape[1]
        num_he_compare = he_compare.shape[1]


        he_anchor = self.a2h(he_anchor)
        if self.ver == 1:
            he_compare = self.c2h(he_compare)
        else:
            he_compare = self.a2h(he_compare)

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
        if self.out_activation == 'sigmoid':
            score = F.sigmoid(score)
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

            # return score_p, score_n, loss
            return None, loss


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


def get_mlp(in_dim, h_dim, out_dim):
    net = nn.Sequential(nn.Linear(in_dim, h_dim),
                        nn.ReLU(),
                        nn.Linear(h_dim, out_dim))
    return net


class GraphEmbedding(nn.Module):
    """Graph (node) embedding algorithms for baseline"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.algo = cfg['MODEL'].get('ALGO', 'GCN')
        # self.out_activation = cfg['MODEL'].get('OUTACT', 'linear')
        self.word_emb_size = cfg['MODEL']['WORD_EMB_SIZE']
        self.n_hidden = cfg['MODEL']['NUM_HIDDEN']
        if self.cfg['MODEL']['TARGET'] == 'SBERT':
            self.loss = nn.MSELoss()
        else:
            self.margin = cfg['MODEL']['LOSS_MARGIN']
            self.loss = TripletLoss(self.margin)

        if self.algo in ('GCN', 'GCN1', 'GCN2', 'GCN3'):
            self.conv1 = gnn.DenseGCNConv(self.word_emb_size, self.n_hidden)
            self.conv2 = gnn.DenseGCNConv(self.n_hidden, self.n_hidden)
            self.conv3 = gnn.DenseGCNConv(self.n_hidden, self.n_hidden)
        elif self.algo in ('GCN4', ):
            self.conv1 = gnn.DenseGCNConv(self.word_emb_size, self.n_hidden)
            self.conv2 = gnn.DenseGCNConv(self.n_hidden, self.n_hidden)
            self.conv3 = gnn.DenseGCNConv(self.n_hidden, self.n_hidden)
            self.conv4 = gnn.DenseGCNConv(self.n_hidden, self.n_hidden)
        elif self.algo in ('GIN'):
            self.mlp_hidden = cfg['MODEL']['MLP_HIDDEN']
            self.nn1 = get_mlp(self.word_emb_size, self.mlp_hidden, self.n_hidden)
            self.nn2 = get_mlp(self.n_hidden, self.mlp_hidden, self.n_hidden)
            self.nn3 = get_mlp(self.n_hidden, self.mlp_hidden, self.n_hidden)
            self.conv1 = gnn.DenseGINConv(self.nn1)
            self.conv2 = gnn.DenseGINConv(self.nn2)
            self.conv3 = gnn.DenseGINConv(self.nn3)
        else:
            raise ValueError

    def _embed(self, graph):
        x = graph['x']
        adj = graph['adj']
        num_node = graph['n_node'].flatten()
        if self.algo == 'GCN' or self.algo == 'GCN3':
            h = self.conv1(x, adj)
            h = F.relu(h)
            h = self.conv2(h, adj)
            h = F.relu(h)
            h = self.conv3(h, adj)
            h = h.sum(dim=1) / num_node[:,None].to(torch.float)  # average pooling
            h = h / h.norm(dim=1, keepdim=True)
            return h
        elif self.algo == 'GCN1':
            h = self.conv1(x, adj)
            h = h.sum(dim=1) / num_node[:,None].to(torch.float)  # average pooling
            h = h / h.norm(dim=1, keepdim=True)
            return h
        elif self.algo == 'GCN2':
            h = self.conv1(x, adj)
            h = F.relu(h)
            h = self.conv2(h, adj)
            h = h.sum(dim=1) / num_node[:,None].to(torch.float)  # average pooling
            h = h / h.norm(dim=1, keepdim=True)
            return h
        elif self.algo == 'GCN4':
            h = self.conv1(x, adj)
            h = F.relu(h)
            h = self.conv2(h, adj)
            h = F.relu(h)
            h = self.conv3(h, adj)
            h = F.relu(h)
            h = self.conv4(h, adj)
            h = h.sum(dim=1) / num_node[:,None].to(torch.float)  # average pooling
            h = h / h.norm(dim=1, keepdim=True)
            return h
        elif self.algo == 'GIN':
            h = self.conv1(x, adj)
            h = F.relu(h)
            h = self.conv2(h, adj)
            h = F.relu(h)
            h = self.conv3(h, adj)
            h = h.sum(dim=1) / num_node[:,None].to(torch.float)  # average pooling
            h = h / h.norm(dim=1, keepdim=True)
            return h

    def score(self, data1, data2, **kwargs):
        emb1 = self._embed(data1)
        emb2 = self._embed(data2)
        score = (emb1 * emb2).sum(dim=1)

        return score, None  # (score, attmap)

    def forward(self, anchor, pos, neg):
        if self.cfg['MODEL']['TARGET'] == 'SBERT':
            # neg is sbert score
            score_bert = neg
            score, _ = self.score(anchor, pos)
            loss = self.loss(score, score_bert)
            return None, loss
        else:
            emb_anchor = self._embed(anchor)
            emb_pos = self._embed(pos)
            emb_neg = self._embed(neg)
            score_p = (emb_anchor * emb_pos).sum(dim=1)
            score_n = (emb_anchor * emb_neg).sum(dim=1)
            loss = self.loss(score_p, score_n)
            return None, loss  # (score, loss)
