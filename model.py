from math import ceil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from torchvision.models import resnet18
import torch_geometric.nn as gnn
from model_ssgpool import l2norm, GNN_Block, GNN_Block_sparse
from model_graph_ssgpool import dense_diff_pool, get_Spectral_loss, dense_ssgpool, dense_ssgpool_gumbel, \
    get_spectral_loss_mini, SAGPooling
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import global_mean_pool


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

            if self.cfg['MODEL']['FEATURE'] == 'adj_he':
                self.gcn = gnn.DenseGCNConv(self.word_emb_size, self.word_emb_size)

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
        elif self.cfg['MODEL']['FEATURE'] == 'adj_he':
            node_feature = x['x']
            adj = x['adj']
            he = x['he']  # Batch x max_HE x step
            node_x = self.gcn(node_feature, adj)  # Batch x node x feature
            he = he.unsqueeze(3).repeat(1,1,1,node_x.shape[-1])  # [B, nHE, step, feature]
            node_x = node_x.unsqueeze(1).repeat(1,he.shape[1],1,1)  # [B, nHE, node, feature]
            # out = torch.zeros(he.shape[0], he.shape[1], node_x.shape[2]).to(node_feature.device)
            out = torch.gather(node_x, 2, he.clamp(min=0))
            mask = he >= 0
            out[~mask] = 0
            return out.sum(2) / mask.sum(2).clamp(min=1.).to(torch.float)
            # print(out.shape)
            # for i_batch in range(he.shape[0]):
            #     for i_he in range(he.shape[1]):
            #         nodes = he[i_batch, i_he]
            #         nodes = nodes[nodes >= 0]  # ignore -1 index
            #         mean_rep = node_x[i_batch, nodes, :].mean(dim=0)
            #         out[i_batch, i_he] = mean_rep
            # return out

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


class HGAN_V0(HGAN):
    def score(self, he_anchor, he_compare):
        he_anchor = self._preprocess(he_anchor)  # [B, #a, d]
        he_compare = self._preprocess(he_compare)  # [B, #p, d]

        anchor = he_anchor.mean(dim=1)
        anchor = anchor / anchor.norm(dim=1, keepdim=True)
        compare = he_compare.mean(dim=1)
        compare = compare / compare.norm(dim=1, keepdim=True)
        return (anchor * compare).sum(dim=1, keepdim=True), None


class HGAN_V2(HGAN):
    def score(self, he_anchor, he_compare):
        he_anchor = self._preprocess(he_anchor)
        he_compare = self._preprocess(he_compare)

        he_anchor = self.a2h(he_anchor)
        he_compare = self.a2h(he_compare)

        # normalize
        # norm_anchor = he_anchor.norm(dim=2, keepdim=True)  # [B, #a, 1]
        # norm_compare = he_compare.norm(dim=2, keepdim=True)  #  [B, #p, 1]
        # nonzero_anchor = (norm_anchor > 0).sum(dim=2).sum(dim=1)  # [B,]
        # nonzero_compare = (norm_compare > 0).sum(dim=2).sum(dim=1)  # [B,]
        # he_anchor = he_anchor / norm_anchor.clamp(min=0.001)
        # he_compare = he_compare / norm_compare.clamp(min=0.001)

        # anchor = he_anchor.sum(dim=1) / nonzero_anchor[:, None].to(torch.float)  # [B, d]
        anchor = he_anchor.mean(dim=1)
        anchor = anchor / anchor.norm(dim=1, keepdim=True)
        # compare = he_compare.sum(dim=1) / nonzero_compare[:, None].to(torch.float)  # [B, d]
        compare = he_compare.mean(dim=1)
        compare = compare / compare.norm(dim=1, keepdim=True)
        return (anchor * compare).sum(dim=1, keepdim=True), None


class HGAN_V3(HGAN):
    def score(self, he_anchor, he_compare):
        he_anchor = self._preprocess(he_anchor)
        he_compare = self._preprocess(he_compare)

        he_anchor = self.a2h(he_anchor)  # [B, #a, d]
        he_compare = self.a2h(he_compare)  # [B, #p, d]

        # normalize
        norm_anchor = he_anchor.norm(dim=2, keepdim=True)  # [B, #a, 1]
        norm_compare = he_compare.norm(dim=2, keepdim=True)  # [B, #p, 1]
        nonzero_anchor = (norm_anchor > 0).sum(dim=2).sum(dim=1).to(torch.float)  # [B,]
        nonzero_compare = (norm_compare > 0).sum(dim=2).sum(dim=1).to(torch.float)  # [B,]
        # he_anchor = he_anchor / norm_anchor.clamp(min=0.001)
        # he_compare = he_compare / norm_compare.clamp(min=0.001)

        # key vector for self-attention
        key_anchor = self.h2att(he_anchor)  # [B, #a, d]
        key_compare = self.h2att(he_compare)  # [B, #p, d]

        self_mul = torch.matmul(key_anchor, key_compare.permute(0, 2, 1))  # [B, #a, #p]
        att = torch.sigmoid(self_mul)
        pair_sim = torch.matmul(he_anchor, he_compare.permute(0, 2, 1))
        score = (pair_sim * att).sum(dim=2).sum(dim=1) / nonzero_anchor / nonzero_compare
        return score[:, None], att


class HGAN_V4(HGAN):
    def score(self, he_anchor, he_compare):
        he_anchor = self._preprocess(he_anchor)
        he_compare = self._preprocess(he_compare)

        he_anchor = self.a2h(he_anchor)  # [B, #a, d]
        he_compare = self.a2h(he_compare)  # [B, #p, d]

        # normalize
        norm_anchor = he_anchor.norm(dim=2, keepdim=True)  # [B, #a, 1]
        norm_compare = he_compare.norm(dim=2, keepdim=True)  # [B, #p, 1]
        nonzero_anchor = (norm_anchor > 0).sum(dim=2).sum(dim=1).to(torch.float)  # [B,]
        nonzero_compare = (norm_compare > 0).sum(dim=2).sum(dim=1).to(torch.float)  # [B,]
        he_anchor = he_anchor / norm_anchor.clamp(min=0.001)
        he_compare = he_compare / norm_compare.clamp(min=0.001)

        # key vector for self-attention
        key_anchor = self.h2att(he_anchor)  # [B, #a, d]
        key_compare = self.h2att(he_compare)  # [B, #p, d]

        self_mul = torch.matmul(key_anchor, key_compare.permute(0, 2, 1))  # [B, #a, #p]
        att_1 = F.softmax(self_mul, dim=1)
        att_2 = F.softmax(self_mul, dim=2)
        att = (att_1 + att_2) / 2
        pair_sim = torch.matmul(he_anchor, he_compare.permute(0, 2, 1))
        score = (pair_sim * att).sum(dim=2).sum(dim=1) / nonzero_anchor / nonzero_compare
        return score[:, None], att


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

        if self.algo in ('GCN', 'GCN1', 'GCN2', 'GCN3', 'GCNA1'):
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

        if self.algo in ('GCNA1',):
            self.nhead = 8
            # self.a2h = torch.nn.Linear(self.word_emb_size, self.nhidden)
            self.h2att = torch.nn.Linear(self.n_hidden, self.nhead)
            self.softmax_att = torch.nn.Softmax(dim=2)
            self.fc_score = torch.nn.Linear(self.n_hidden*self.nhead, 1)


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

    def _embed_node(self, graph):
        x = graph['x']
        adj = graph['adj']
        num_node = graph['n_node'].flatten()
        h = self.conv1(x, adj)
        h = F.relu(h)
        h = self.conv2(h, adj)
        h = F.relu(h)
        h = self.conv3(h, adj)
        return h

    def score(self, data1, data2, **kwargs):
        if self.algo in ('GCNA1',):
            he_anchor = self._embed_node(data1)
            he_compare = self._embed_node(data2)
            num_node1 = data1['n_node'].flatten()
            num_node2 = data2['n_node'].flatten()
            num_he_anchor = he_anchor.shape[1]
            num_he_compare = he_compare.shape[1]

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

            he_anchor = he_anchor.permute(0, 2, 1)  # [B, #a, d]
            pairwise = torch.matmul(he_anchor, he_compare)  # [B, #a, #p]

            l_out = []
            for i in range(self.nhead):
                att_g = att_map[:, i, :, :]  # [B, #a, #p]
                sim = (pairwise * att_g).sum(dim=2).sum(dim=1)
                l_out.append(sim)
            out = torch.stack(l_out, dim=1)
            score = out.mean(dim=1)
            return score, att_map
        else:
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



class GraphPool(nn.Module):
    """Graph pooling"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.algo = cfg['MODEL'].get('ALGO', 'DiffPool')
        self.word_emb_size = cfg['MODEL']['WORD_EMB_SIZE']
        self.n_hidden = cfg['MODEL']['NUM_HIDDEN']
        self.n_pooled_node = cfg['MODEL'].get('NUM_POOLED', 200)
        if self.cfg['MODEL']['TARGET'] == 'SBERT':
            self.mseloss = nn.MSELoss()
        else:
            self.margin = cfg['MODEL']['LOSS_MARGIN']
            self.loss = TripletLoss(self.margin)

        if self.algo in ('DiffPool',):
            self.num_layers = cfg['MODEL']['N_LAYERS']
            pool_ratio = cfg['MODEL']['POOL_RATIO']
            max_num_nodes = 300
            num_nodes = ceil(pool_ratio * max_num_nodes)
            self.embed_block1 = GNN_Block(self.word_emb_size, self.n_hidden,
                                          self.n_hidden)  # DenseGCNConv(word_dim, embed_size) #
            self.pool_block1 = GNN_Block(self.word_emb_size, self.n_hidden,
                                         num_nodes)  # DenseGCNConv(embed_size, num_nodes) #
            self.embed_blocks = torch.nn.ModuleList()
            self.pool_blocks = torch.nn.ModuleList()
            for i in range(self.num_layers - 1):
                num_nodes = ceil(pool_ratio * num_nodes)
                self.embed_blocks.append(GNN_Block(self.n_hidden, self.n_hidden, self.n_hidden))
                self.pool_blocks.append(GNN_Block(self.n_hidden, self.n_hidden, num_nodes))

            self.embed_final = GNN_Block(self.n_hidden, self.n_hidden,
                                         self.n_hidden)  # DenseGCNConv(embed_size, embed_size) #
            self.linear_f = nn.Linear(self.n_hidden * (self.num_layers + 1), self.n_hidden)

        elif self.algo == 'SSGPool':
            self.gnn_embed = GNN_Block(self.word_emb_size, self.n_hidden, self.n_hidden) #DenseGCNConv(word_dim, embed_size) # #
            self.gnn_pool = GNN_Block(self.word_emb_size, self.n_hidden, self.n_pooled_node) #DenseGCNConv(embed_size, 20) # #
            self.gnn_embed_f = GNN_Block(self.n_hidden, self.n_hidden, self.n_hidden)#DenseGCNConv(embed_size, embed_size)#
        elif self.algo == 'SSGPoolV2':
            self.num_layers = cfg['MODEL']['N_LAYERS']
            pool_ratio = cfg['MODEL']['POOL_RATIO']
            max_num_nodes = 300
            num_nodes = ceil(pool_ratio * max_num_nodes)
            self.embed_block1 = GNN_Block(self.word_emb_size, self.n_hidden, self.n_hidden) #DenseGCNConv(word_dim, embed_size) #
            self.pool_block1 = GNN_Block(self.word_emb_size, self.n_hidden, num_nodes) #DenseGCNConv(embed_size, num_nodes) #
            self.embed_blocks = torch.nn.ModuleList()
            self.pool_blocks = torch.nn.ModuleList()
            for i in range(self.num_layers - 1):
                num_nodes = ceil(pool_ratio * num_nodes)
                self.embed_blocks.append(GNN_Block(self.n_hidden, self.n_hidden, self.n_hidden))
                self.pool_blocks.append(GNN_Block(self.n_hidden, self.n_hidden, num_nodes))

            self.embed_final = GNN_Block(self.n_hidden, self.n_hidden, self.n_hidden) #DenseGCNConv(embed_size, embed_size) #
            self.linear_f = nn.Linear(self.n_hidden * (self.num_layers + 1) , self.n_hidden)
        elif self.algo == 'SAGPool':
            self.gnn_embed = GNN_Block_sparse(self.word_emb_size, self.n_hidden, self.n_hidden) #GCNConv(word_dim, embed_size) #
            pool_ratio = cfg['MODEL']['POOL_RATIO']
            self.gnn_pool = SAGPooling(self.n_hidden, pool_ratio)
            self.gnn_embed_f = GNN_Block_sparse(self.n_hidden, self.n_hidden, self.n_hidden) #GCNConv(embed_size, embed_size) #
            self.linear_f = nn.Linear(self.n_hidden * 2 , self.n_hidden)
        elif self.algo == 'NoPool':
            self.num_layers = cfg['MODEL']['N_LAYERS']
            pool_ratio = cfg['MODEL']['POOL_RATIO']
            max_num_nodes = 300
            num_nodes = ceil(pool_ratio * max_num_nodes)
            self.embed_block1 = GNN_Block(self.word_emb_size, self.n_hidden,
                                          self.n_hidden)  # DenseGCNConv(word_dim, embed_size) #
            self.embed_blocks = torch.nn.ModuleList()
            for i in range(self.num_layers - 1):
                num_nodes = ceil(pool_ratio * num_nodes)
                self.embed_blocks.append(GNN_Block(self.n_hidden, self.n_hidden, self.n_hidden))

            self.embed_final = GNN_Block(self.n_hidden, self.n_hidden,
                                         self.n_hidden)  # DenseGCNConv(embed_size, embed_size) #
            self.linear_f = nn.Linear(self.n_hidden * (self.num_layers + 1), self.n_hidden)

        else:
            raise ValueError

        self.lambda_reg = self.cfg['MODEL'].get('LAMBDA_REG', 0)
        self.use_specloss = self.cfg['MODEL'].get('SPEC_LOSS', False)
        self.use_entrloss = self.cfg['MODEL'].get('ENTR_LOSS', False)
        self.use_linkloss = self.cfg['MODEL'].get('LINK_LOSS', False)

    def _embed(self, graph):
        x = graph['x']
        adj = graph['adj']
        lengths = graph['n_node'].flatten()
        if self.algo == 'DiffPool':

            spec_losses = 0.
            entr_losses = 0.
            link_losses = 0.
            mask = torch.arange(max(lengths)).expand(len(lengths), max(lengths)).cuda() < lengths.unsqueeze(1)
            mask = mask.to(torch.float)
            #x = self.embed(x)

            s = self.pool_block1(x, adj, mask, add_loop=True)
            x = F.relu(self.embed_block1(x, adj, mask, add_loop=True))
            xs = [torch.sum(x, 1) / (mask.sum(-1, keepdims=True).to(x.dtype) + 1e-10)]
            x, adj, link_loss, ent_loss = dense_diff_pool(x, adj, s, mask)
            link_losses += link_loss
            entr_losses += ent_loss
            for i, (embed_block, pool_block) in enumerate(
                    zip(self.embed_blocks, self.pool_blocks)):
                s = pool_block(x, adj)
                x = F.relu(embed_block(x, adj))
                xs.append(x.mean(dim=1))
                if i < len(self.embed_blocks):
                    x, adj, link_loss, ent_loss = dense_diff_pool(x, adj, s)
                    link_losses += link_loss
                    entr_losses += ent_loss

            spec_losses += torch.Tensor([0.])
            # features = F.dropout(features, 0.5)
            x = self.embed_final(x, adj)

            xs.append(torch.mean(x, 1))

            feature_out = self.linear_f(torch.cat(xs, -1))

            feature_out = l2norm(feature_out)
            return feature_out, (spec_losses, entr_losses, link_losses)
        elif self.algo == 'SSGPool':
            spec_losses = 0.
            entr_losses = 0.
            coarsen_losses = 0.
            mask = torch.arange(max(lengths)).expand(len(lengths), max(lengths)).cuda() < lengths.unsqueeze(1)
            mask = mask.to(torch.float)

            B, N, _ = adj.size()
            s_inv_final = torch.eye(N).unsqueeze(0).expand(B, N, N).cuda()
            s_final = torch.eye(N).unsqueeze(0).expand(B, N, N).cuda()

            x_next = F.relu(self.gnn_embed(x, adj, mask))
            s = self.gnn_pool(x, adj, mask)
            x_next, a_next, Lapl, L_next, s, s_inv = dense_ssgpool(x_next, adj, s, mask)

            s_final = torch.bmm(s_final, s)

            '''Second layer'''
            x = self.gnn_embed_f(x_next, a_next)
            feature_out = x.mean(dim=1)

            s_inv_final = s_final.transpose(1, 2) / ((s_final * s_final).sum(dim=1).unsqueeze(
                -1) + 1e-10)

            spec_loss = get_Spectral_loss(Lapl, L_next, s_inv_final.transpose(1, 2), 3, mask)

            coarsen_loss = adj - torch.matmul(s_final, s_final.transpose(1,2))
            mask_ = mask.view(B, N, 1).to(x.dtype)
            coarsen_loss = coarsen_loss * mask_
            coarsen_loss = coarsen_loss * mask_.transpose(1, 2)
            coarsen_loss = torch.sqrt((coarsen_loss * coarsen_loss).sum(dim=(1, 2)))

            spec_losses += spec_loss.mean()
            entr_losses += torch.Tensor([0.]) #entr_loss.mean()
            coarsen_losses += coarsen_loss.mean()

            feature_out = l2norm(feature_out)

            return feature_out, (spec_losses, entr_losses, coarsen_losses)
        elif self.algo == 'SSGPoolV2':
            spec_losses = 0.
            spec_losses_hard = 0.
            entr_losses = 0.
            coarsen_losses = 0.

            mask = torch.arange(max(lengths)).expand(len(lengths), max(lengths)).cuda() < lengths.unsqueeze(1)
            mask = mask.to(torch.float)
            B, N, _ = adj.size()
            s_final = torch.eye(N).unsqueeze(0).expand(B, N, N).cuda()
            s_inv_final = torch.eye(N).unsqueeze(0).expand(B, N, N).cuda()
            s_inv_soft_final = torch.eye(N).unsqueeze(0).expand(B, N, N).cuda()
            ori_adj = adj

            # x = self.embed(x)
            s = self.pool_block1(x, adj, mask, add_loop=True)
            x = F.relu(self.embed_block1(x, adj, mask, add_loop=True))
            xs = [torch.sum(x, 1) / (mask.sum(-1, keepdims=True).to(x.dtype) + 1e-10)]

            diag_ele = torch.sum(adj, -1)
            Diag = torch.diag_embed(diag_ele)
            Lapl = Diag - adj
            Lapl_ori = Lapl
            x_before = x
            L_before = Lapl
            x, adj, L_next, L_next_soft, s, s_soft, s_inv, s_inv_soft = dense_ssgpool_gumbel(x, adj, s, Lapl, Lapl, mask, is_training=self.training)
            spec_loss, spec_loss_soft = get_spectral_loss_mini(x_before, s, s_inv, s_soft, s_inv_soft, L_before, mask)
            spec_losses +=spec_loss_soft
            spec_losses_hard += spec_loss

            s_final = torch.bmm(s_final, s)
            s_inv_final = torch.bmm(s_inv, s_inv_final)
            s_inv_soft_final = torch.bmm(s_inv_soft, s_inv_soft_final)

            for i, (embed_block, pool_block) in enumerate(
                    zip(self.embed_blocks, self.pool_blocks)):
                s = pool_block(x, adj, add_loop=True)
                x = F.relu(embed_block(x, adj, add_loop=True))
                xs.append(x.mean(dim=1))
                if i < len(self.embed_blocks):
                    x_before = x
                    L_before = L_next
                    x, adj, L_next, L_next_soft, s, s_soft, s_inv, s_inv_soft = dense_ssgpool_gumbel(x, adj, s, L_next, L_next_soft, is_training=self.training)
                    spec_loss, spec_loss_soft = get_spectral_loss_mini(x_before, s, s_inv, s_soft, s_inv_soft, L_before, mask)
                    spec_losses += spec_loss_soft
                    spec_losses_hard += spec_loss
                    s_final = torch.bmm(s_final, s)
                    s_inv_final = torch.bmm(s_inv, s_inv_final)
                    s_inv_soft_final = torch.bmm(s_inv_soft, s_inv_soft_final)

            x = self.embed_final(x, adj, add_loop=True)
            xs.append(x.mean(dim=1))

            #feature_out = x.mean(dim=1)
            feature_out = self.linear_f(torch.cat(xs, -1))
            #if feature_out.requires_grad ==True:
            #    #spec_loss, spec_loss_hard = get_Spectral_loss(Lapl_ori, L_next, s_inv_final.transpose(1,2), L_next_soft, s_inv_soft_final.transpose(1, 2), 1, mask)
            #    spec_loss = torch.Tensor([0.])
            #   spec_loss_hard = torch.Tensor([0.])
            #else:
            #    spec_loss = torch.Tensor([0.])
            #    spec_loss_hard = torch.Tensor([0.])
            coarsen_loss = ori_adj - torch.matmul(s_final, s_final.transpose(1,2))
            mask_ = mask.view(B, N, 1).to(x.dtype)
            coarsen_loss = coarsen_loss * mask_
            coarsen_loss = coarsen_loss * mask_.transpose(1, 2)
            coarsen_loss = torch.sqrt((coarsen_loss * coarsen_loss).sum(dim=(1, 2)))


            spec_losses = (spec_losses/self.num_layers).mean()
            spec_losses_hard = (spec_losses_hard/self.num_layers).mean()
            #spec_losses += spec_loss
            #spec_losses_hard += spec_loss_hard
            entr_losses += torch.Tensor([0.]) #entr_loss.mean()
            coarsen_losses += coarsen_loss.mean()

            feature_out = l2norm(feature_out)

            return feature_out, (spec_losses, spec_losses_hard, coarsen_losses)
        elif self.algo == 'SAGPool':
            x = graph['nodes_flat']
            adj = graph['adj_flat']
            batch = graph['batch']
            adj = ((adj + adj.transpose(0, 1)) > 0.).float()
            edge_index, _ = dense_to_sparse(adj)

            # x = self.embed(x)
            x = F.relu(self.gnn_embed(x, edge_index))
            xs = [global_mean_pool(x, batch)]
            x, edge_index, _, batch, _, _ = self.gnn_pool(x, edge_index, batch=batch)
            x = self.gnn_embed_f(x, edge_index)

            xs.append(global_mean_pool(x, batch))
            feature_out = self.linear_f(torch.cat(xs, -1))
            feature_out = l2norm(feature_out)

            spec_losses = torch.Tensor([0.])
            entr_losses = torch.Tensor([0.])
            coarsen_losses = torch.Tensor([0.])
            return feature_out, (spec_losses, entr_losses, coarsen_losses)
        elif self.algo == 'NoPool':
            mask = torch.arange(max(lengths)).expand(len(lengths), max(lengths)).cuda() < lengths.unsqueeze(1)
            mask = mask.to(torch.float)

            x = F.relu(self.embed_block1(x, adj, mask, add_loop=True))
            xs = [torch.sum(x, 1) / (mask.sum(-1, keepdims=True).to(x.dtype) + 1e-10)]
            x = self.embed_final(x, adj)

            xs.append(torch.mean(x, 1))

            feature_out = self.linear_f(torch.cat(xs, -1))

            feature_out = l2norm(feature_out)

            spec_losses = torch.Tensor([0.])
            entr_losses = torch.Tensor([0.])
            link_losses = torch.Tensor([0.])
            coarsen_losses = torch.Tensor([0.])
            return feature_out, (spec_losses, entr_losses, link_losses)


    def score(self, data1, data2, **kwargs):
        emb1, reg_loss1 = self._embed(data1)
        emb2, reg_loss2 = self._embed(data2)
        score = (emb1 * emb2).sum(dim=1)

        return score, [reg_loss1, reg_loss2]  # (score, pooling losses)

    def forward(self, anchor, pos, neg):
        if self.cfg['MODEL']['TARGET'] == 'SBERT':
            # neg is sbert score
            score_bert = neg
            score, l_loss = self.score(anchor, pos)
            loss = self.loss(score, score_bert, l_loss)
            return None, loss 
        else:
            emb_anchor = self._embed(anchor)
            emb_pos = self._embed(pos)
            emb_neg = self._embed(neg)
            score_p = (emb_anchor * emb_pos).sum(dim=1)
            score_n = (emb_anchor * emb_neg).sum(dim=1)
            loss = self.loss(score_p, score_n)
            return None, loss  # (score, loss)

    def loss(self, pred, target, l_reg_loss):
        loss = self.mseloss(pred, target)
        reg_loss1, reg_loss2 = l_reg_loss  # losses for each images
        specloss = torch.tensor(0.).to(pred.device)
        entrloss = torch.tensor(0.).to(pred.device)
        linkloss = torch.tensor(0.).to(pred.device)

        for i, (reg1, reg2) in enumerate(zip(reg_loss1, reg_loss2)):
            if i == 0:
                specloss = (reg1 + reg2).mean()
                if self.use_specloss:
                    loss += self.lambda_reg * specloss
            if i == 1:
                entrloss = (reg1 + reg2).mean()
                if self.use_entrloss:
                    loss += self.lambda_reg * entrloss
            if i == 2:
                linkloss = (reg1 + reg2).mean()
                if self.use_linkloss:
                    loss += self.lambda_reg * linkloss
        return {'loss': loss, 'specloss': specloss, 'entrloss': entrloss, 'linkloss': linkloss}
