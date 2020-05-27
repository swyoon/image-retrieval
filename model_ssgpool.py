from math import ceil
import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import  DenseSAGEConv, DenseGINConv, GCNConv, global_mean_pool
from torch_geometric.utils import dense_to_sparse
from model_graph_ssgpool import DenseGCNConv, dense_diff_pool, get_Spectral_loss, dense_ssgpool, dense_ssgpool_gumbel, SAGPooling, dense_ssgpool_gumbel_select
#from torch_geometric.nn import GCNConv

from sentence_transformers import SentenceTransformer


def l2norm(X, eps=1e-10):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm+eps)
    return X

class GNN_Block(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN_Block, self).__init__()

        self.conv1 = DenseGCNConv(in_channels, hidden_channels)
        self.conv2 = DenseGCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels + hidden_channels, out_channels)


    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, adj, mask=None, add_loop=True):
        x1 = F.relu(self.conv1(x, adj, mask, add_loop))
        x2 = F.relu(self.conv2(x1, adj, mask, add_loop))
        out = self.lin(torch.cat((x1, x2), -1))

        return out


class GNN_Block_sparse(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN_Block_sparse, self).__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels + hidden_channels, out_channels)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, edge_index, mask=None, add_loop=True):
        x1 = F.relu(self.conv1(x, edge_index))
        x2 = F.relu(self.conv2(x1, edge_index))
        out = self.lin(torch.cat((x1, x2), -1))

        return out

class Encoder_word(nn.Module):
    def __init__(self, vocab_size, word_dim, word_emb=None, dropRate=0.0):
        super(Encoder_word, self).__init__()

        self.embed = nn.Embedding(vocab_size, word_dim)
        self.embed.weight.requires_grad = False
        self.word_emb = word_emb
        self.droprate = dropRate
        self.init_weights()

    def init_weights(self):
        if self.word_emb is not None:
            self.embed.weight.data.copy_(torch.from_numpy(self.word_emb))
        else:
            self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, X, eps=1e-10):
        out = self.embed(X)

        return out

def EncoderImage(data_name, img_dim, vocab_size, word_dim, embed_size,  word_emb, finetune=False,
                 cnn_type='vgg19', gmodel_type=None, use_abs=False, no_imgnorm=False, use_SG=False,
                 use_specloss=False, use_entrloss=False, use_linkloss=False, num_layers=1, pool_ratio=0.1):
    """A wrapper to image encoders. Chooses between an encoder that uses
    precomputed image features, `EncoderImagePrecomp`, or an encoder that
    computes image features on the fly `EncoderImageFull`.
    """
    if data_name.endswith('_precomp'):
        img_enc = EncoderImagePrecomp(
            img_dim, embed_size, use_abs, no_imgnorm)
    else:
        '''
        if use_SG == True and cnn_type is not None:
            if gmodel_type=='diffpool':
                img_enc = EncoderSSGPool_select_cnn(vocab_size, word_dim,
                                          embed_size, word_emb, use_abs, no_imgnorm, finetune, cnn_type,
                                          use_specloss, use_entrloss, use_linkloss)
            elif gmodel_type=='ssgpool':
                img_enc = EncoderSSGPool_select_cnn(vocab_size, word_dim,
                                          embed_size, word_emb, use_abs, no_imgnorm, finetune, cnn_type,
                                          use_specloss, use_entrloss, use_linkloss)

            elif gmodel_type=='ssgpool_select':
                img_enc = EncoderSSGPool_select_cnn(vocab_size, word_dim,
                                         embed_size, word_emb, use_abs, no_imgnorm, finetune, cnn_type,
                                         use_specloss, use_entrloss, use_linkloss)
            elif gmodel_type=='gcn':
                img_enc = EncoderSSGPool_select_cnn(vocab_size, word_dim,
                                         embed_size, word_emb, use_abs, no_imgnorm, finetune, cnn_type,
                                         use_specloss, use_entrloss, use_linkloss)
            else:
                print("gmodel_type NEEDED")
        '''
        if use_SG == True:
            if gmodel_type=='diffpool':
                img_enc = EncoderDiffPool(vocab_size, word_dim,
                                          embed_size, word_emb, use_abs, no_imgnorm,
                                          use_specloss, use_entrloss, use_linkloss, num_layers, pool_ratio)
            elif gmodel_type=='ssgpool':
                img_enc = EncoderSSGPool(vocab_size, word_dim,
                                          embed_size, word_emb, use_abs, no_imgnorm,
                                          use_specloss, use_entrloss, use_linkloss, num_layers, pool_ratio)

            elif gmodel_type=='gcn':
                img_enc = EncoderGCN(vocab_size, word_dim,
                                         embed_size, word_emb, use_abs, no_imgnorm, num_layers)
            elif gmodel_type=='sagpool':
                img_enc = EncoderSAGPool(vocab_size, word_dim,
                                          embed_size, word_emb, use_abs, no_imgnorm,
                                          use_specloss, use_entrloss, use_linkloss, num_layers, pool_ratio)

            else:
                print("gmodel_type NEEDED")
        else:
            img_enc = EncoderImageFull(
                embed_size, finetune, cnn_type, use_abs, no_imgnorm)

    return img_enc

class EncoderGCN(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, word_emb, use_abs=False, no_imgnorm=False, num_layers=1):
        """Load pretrained VGG19 and replace top fc layer."""
        super(EncoderGCN, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs
        self.embed = word_emb
        self.embed_final = nn.Linear(embed_size, embed_size)

        # Load a pre-trained model
        self.gnn1 = GNN_Block(word_dim, embed_size, embed_size) # DenseGCNConv(word_dim, embed_size)
        self.gnn2 = GNN_Block(embed_size, embed_size, embed_size) # DenseGCNConv(embed_size, embed_size)
        #self.gnn3 = DenseGCNConv(embed_size, embed_size)

    def forward(self, images, lengths):
        """Extract image feature vectors."""
        x = images['nodes']
        adj = images['adj']
        adj = ((adj + adj.transpose(1, 2)) > 0.).float()

        mask = torch.arange(max(lengths)).expand(len(lengths), max(lengths)).cuda() < lengths.unsqueeze(1)

        x = self.embed(x)
        features = F.relu(self.gnn1(x, adj, mask))
        #features = F.dropout(features, 0.5)
        features = self.gnn2(features, adj, mask)
        #features = self.gnn3(features, adj, mask)
        #features += x


        feature_out = torch.sum(features, 1) / lengths.unsqueeze(-1).to(x.dtype)
        #feature_out = self.embed_final(feature_out)
        #feature_out = self.embed_final(feature_out)

        if not self.no_imgnorm:
            feature_out = l2norm(feature_out)

        # take the absolute value of the embedding (used in order embeddings)
        if self.use_abs:
            feature_out = torch.abs(feature_out)
        #feature_out = F.dropout(feature_out, 0.5)
        regloss = (torch.Tensor([0.]), torch.Tensor([0.]), torch.Tensor([0.]))#0.
        return feature_out, regloss

class EncoderDiffPool(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, word_emb, use_abs=False, no_imgnorm=False,
                 use_specloss=False, use_entrloss=False, use_linkloss=False, num_layers=1, pool_ratio=0.1):
        """Load pretrained VGG19 and replace top fc layer."""
        super(EncoderDiffPool, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs
        self.use_specloss = use_specloss
        self.use_entrloss = use_entrloss
        self.use_linkloss = use_linkloss
        self.embed = word_emb
        #self.embed_final = nn.Linear(embed_size, embed_size)

        num_layers = num_layers
        ratio = pool_ratio
        max_num_nodes = 300
        num_nodes = num_nodes = ceil(ratio * max_num_nodes)
        # Load a pre-trained model
        self.embed_block1 = GNN_Block(word_dim, embed_size, embed_size) #DenseGCNConv(word_dim, embed_size)  #
        # self.gnn_pool = GNN_Block(embed_size, embed_size, 1) #DenseGCNConv(embed_size, 20)#
        self.pool_block1 = GNN_Block(embed_size, embed_size, num_nodes) #DenseGCNConv(embed_size, num_nodes)  #

        self.embed_blocks = torch.nn.ModuleList()
        self.pool_blocks = torch.nn.ModuleList()

        for i in range(num_layers - 1):
            num_nodes = ceil(ratio * num_nodes)
            self.embed_blocks.append(GNN_Block(embed_size, embed_size, embed_size))
            self.pool_blocks.append(GNN_Block(embed_size, embed_size, num_nodes))

        self.embed_final = GNN_Block(embed_size, embed_size, embed_size)  # GNN_Block(embed_size, embed_size, embed_size) #
        self.linear_f = nn.Linear(embed_size * (num_layers+1), embed_size)
    def forward(self, images, lengths):
        """Extract image feature vectors."""
        x = images['nodes']
        adj = images['adj']
        adj = ((adj + adj.transpose(1, 2)) > 0.).float()
        spec_losses = 0.
        entr_losses = 0.
        link_losses = 0.
        mask = torch.arange(max(lengths)).expand(len(lengths), max(lengths)).cuda() < lengths.unsqueeze(1)

        x = self.embed(x)

        s = self.pool_block1(x, adj, mask, add_loop=True)
        x = F.relu(self.embed_block1(x, adj, mask, add_loop=True))
        xs = [torch.sum(x, 1) / (mask.sum(-1, keepdims=True).to(x.dtype)+1e-10)]
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
        #features = F.dropout(features, 0.5)
        x = self.embed_final(x, adj)

        xs.append(torch.mean(x, 1))
        final_x = F.dropout(torch.cat(xs, -1), p=0.5, training=self.training)
        feature_out = self.linear_f(final_x)
        #feature_out = x.mean(dim=1)



        if not self.no_imgnorm:
            feature_out = l2norm(feature_out)

        # take the absolute value of the embedding (used in order embeddings)
        if self.use_abs:
            feature_out = torch.abs(feature_out)
        #feature_out = F.dropout(feature_out, 0.5)
        return feature_out, (spec_losses, entr_losses, link_losses)


class EncoderSSGPool(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, word_emb, use_abs=False, no_imgnorm=False,
                 use_specloss=False, use_entrloss=False, use_linkloss=False, num_layers=1, pool_ratio=0.1):
        """Load pretrained VGG19 and replace top fc layer."""
        super(EncoderSSGPool, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs
        self.use_specloss = use_specloss
        self.use_entrloss = use_entrloss
        self.use_linkloss = use_linkloss
        self.embed = word_emb
        #self.embed_final = nn.Linear(embed_size, embed_size)
        num_layers = num_layers
        ratio = pool_ratio
        max_num_nodes = 300
        num_nodes = ceil(ratio * max_num_nodes)
        # Load a pre-trained model
        self.embed_block1 = GNN_Block(word_dim, embed_size, embed_size) #DenseGCNConv(word_dim, embed_size) #
        #self.gnn_pool = GNN_Block(embed_size, embed_size, 1) #DenseGCNConv(embed_size, 20)#
        self.pool_block1 = GNN_Block(embed_size, embed_size, num_nodes) #DenseGCNConv(embed_size, num_nodes) #

        self.embed_blocks = torch.nn.ModuleList()
        self.pool_blocks = torch.nn.ModuleList()

        for i in range(num_layers - 1):
            num_nodes = ceil(ratio * num_nodes)
            self.embed_blocks.append(GNN_Block(embed_size, embed_size, embed_size))
            self.pool_blocks.append(GNN_Block(embed_size, embed_size, num_nodes))

        self.embed_final = GNN_Block(embed_size, embed_size, embed_size) #DenseGCNConv(embed_size, embed_size) #
        self.linear_f = nn.Linear(embed_size * (num_layers+1) , embed_size)

    def forward(self, images, lengths):
        """Extract image feature vectors."""
        x = images['nodes']
        adj = images['adj']
        #adj = adj + adj.transpose(1, 2)
        adj = ((adj + adj.transpose(1, 2)) > 0.).float()
        spec_losses = 0.
        spec_losses_hard = 0.
        entr_losses = 0.
        coarsen_losses = 0.
        mask = torch.arange(max(lengths)).expand(len(lengths), max(lengths)).cuda() < lengths.unsqueeze(1)
        B, N, _ = adj.size()
        s_final = torch.eye(N).unsqueeze(0).expand(B, N, N).cuda()
        s_inv_final = torch.eye(N).unsqueeze(0).expand(B, N, N).cuda()
        s_inv_soft_final = torch.eye(N).unsqueeze(0).expand(B, N, N).cuda()
        ori_adj = adj

        x = self.embed(x)
        s = self.pool_block1(x, adj, mask, add_loop=True)
        x = F.relu(self.embed_block1(x, adj, mask, add_loop=True))
        #xs = [torch.sum(x, 1) / (mask.sum(-1, keepdims=True).to(x.dtype) + 1e-10)]
        #x, adj, Lapl, L_next, s, s_inv = dense_ssgpool(x, adj, s, mask)
        diag_ele = torch.sum(adj, -1)
        Diag = torch.diag_embed(diag_ele)
        Lapl = Diag - adj
        Lapl_ori = Lapl
        x, adj, L_next, L_next_soft, s, s_inv, s_inv_soft = dense_ssgpool_gumbel(x, adj, s, Lapl, Lapl, mask, is_training=self.training)

        s_final = torch.bmm(s_final, s)
        s_inv_final = torch.bmm(s_inv, s_inv_final)
        s_inv_soft_final = torch.bmm(s_inv_soft, s_inv_soft_final)

        for i, (embed_block, pool_block) in enumerate(
                zip(self.embed_blocks, self.pool_blocks)):
            s = pool_block(x, adj, add_loop=True)
            x = F.relu(embed_block(x, adj, add_loop=True))
            #xs.append(x.mean(dim=1))
            if i < len(self.embed_blocks):
                #x, adj, _, L_next, s, s_inv = dense_ssgpool(x, adj, s)
                x, adj, L_next, L_next_soft, s, s_inv, s_inv_soft = dense_ssgpool_gumbel(x, adj, s, L_next, L_next_soft, is_training=self.training)
                s_final = torch.bmm(s_final, s)
                s_inv_final = torch.bmm(s_inv, s_inv_final)
                s_inv_soft_final = torch.bmm(s_inv_soft, s_inv_soft_final)

        x = self.embed_final(x, adj, add_loop=True)
        # xs.append(torch.sum(x, 1) / mask.sum(-1, keepdims=True).to(x.dtype))
        #xs.append(torch.mean(x, 1))

        #feature_out = x.sum(dim=1)
        feature_out = x.mean(dim=1)
        #final_x = F.dropout(torch.cat(xs, -1), p=0.5, training=self.training)
        #feature_out = self.linear_f(torch.cat(xs, -1))


        spec_loss, spec_loss_hard = get_Spectral_loss(Lapl_ori, L_next, s_inv_final.transpose(1,2), L_next_soft, s_inv_soft_final.transpose(1, 2), 1, mask)

        '''
        sm_N = s_final.size(-1)
        identity = torch.eye(sm_N).unsqueeze(0).expand(B, sm_N, sm_N).cuda()
        norm_s_final = s_final / torch.sqrt(((s_final * s_final).sum(dim=1,keepdim=True)+ 1e-10))
        coarsen_loss = identity - torch.matmul(norm_s_final.transpose(1,2), norm_s_final)
        coarsen_loss = torch.sqrt((coarsen_loss * coarsen_loss).sum(dim=(1, 2)))
        '''
        coarsen_loss = ori_adj - torch.matmul(s_final, s_final.transpose(1,2))
        mask_ = mask.view(B, N, 1).to(x.dtype)
        coarsen_loss = coarsen_loss * mask_
        coarsen_loss = coarsen_loss * mask_.transpose(1, 2)
        #link_loss = torch.sqrt((link_loss * link_loss).sum(dim=(1, 2))) / (lengths*lengths + 1e-9)
        coarsen_loss = torch.sqrt((coarsen_loss * coarsen_loss).sum(dim=(1, 2)))


        spec_losses += spec_loss.mean()
        spec_losses_hard += spec_loss_hard.mean()
        entr_losses += torch.Tensor([0.]) #entr_loss.mean()
        coarsen_losses += coarsen_loss.mean()


        if not self.no_imgnorm:
            feature_out = l2norm(feature_out)

        # take the absolute value of the embedding (used in order embeddings)
        if self.use_abs:
            feature_out = torch.abs(feature_out)
        #feature_out = F.dropout(feature_out, 0.5)
        return feature_out, (spec_losses, spec_losses_hard, coarsen_losses)

    def forward_vis(self, images, lengths):
        """Extract image feature vectors."""
        x = images['nodes']
        adj = images['adj']
        # adj = adj + adj.transpose(1, 2)
        adj = ((adj + adj.transpose(1, 2)) > 0.).float()
        spec_losses = 0.
        entr_losses = 0.
        coarsen_losses = 0.
        mask = torch.arange(max(lengths)).expand(len(lengths), max(lengths)).cuda() < lengths.unsqueeze(1)
        B, N, _ = adj.size()
        s_inv_final = torch.eye(N).unsqueeze(0).expand(B, N, N).cuda()
        s_final = torch.eye(N).unsqueeze(0).expand(B, N, N).cuda()

        x = self.embed(x)
        x_next = F.relu(self.gnn_embed(x, adj, mask))
        s = self.gnn_pool(x, adj, mask)
        # xs = [x_next.mean(dim=1)]
        # xs = [torch.sum(x, 1) / lengths.unsqueeze(-1).to(x.dtype)]

        x_next, a_next, Lapl, L_next, s, s_inv = dense_ssgpool_gumbel(x_next, adj, s, mask)
        # x_next, a_next, Lapl, L_next, s, s_inv = dense_ssgpool(x_next, adj, s, mask)
        s_final = torch.bmm(s_final, s)
        s_inv_final = torch.bmm(s_inv, s_inv_final)

        '''Second layer'''
        # x = x_next
        # x_next = F.relu(self.gnn_embed2(x, a_next))
        # xs.append(x_next.mean(dim=1))

        # s = self.gnn_pool2(x, a_next)
        # x_next, a_next, _, L_next, s, s_inv = dense_ssgpool(x_next, a_next, s)
        # s_final = torch.bmm(s_final, s)
        # s_inv_final = torch.bmm(s_inv, s_inv_final)

        x = self.gnn_embed_f(x_next, a_next)
        feature_out = x.mean(dim=1)




        if not self.no_imgnorm:
            feature_out = l2norm(feature_out)

        # take the absolute value of the embedding (used in order embeddings)
        if self.use_abs:
            feature_out = torch.abs(feature_out)
        #feature_out = F.dropout(feature_out, 0.5)
        return feature_out, s_final.transpose(1,2)



class EncoderSAGPool(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, word_emb, use_abs=False, no_imgnorm=False,
                 use_specloss=False, use_entrloss=False, use_linkloss=False, num_layers=1, pool_ratio=0.1):
        """Load pretrained VGG19 and replace top fc layer."""
        super(EncoderSAGPool, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs

        self.embed = word_emb
        self.word_dim = word_dim
        self.embed_size = embed_size
        self.gnn_embed = GNN_Block_sparse(word_dim, embed_size, embed_size) #GCNConv(word_dim, embed_size) #
        self.gnn_pool = SAGPooling(embed_size, 0.2)

        #self.gnn_embed2 = DenseGCNConv(word_dim, embed_size) #GNN_Block(word_dim, embed_size, embed_size)  # #
        #self.gnn_pool2 = DenseGCNConv(embed_size, 10) #GNN_Block(embed_size, embed_size, 5) #

        self.gnn_embed_f = GNN_Block_sparse(embed_size, embed_size, embed_size) #GCNConv(embed_size, embed_size) #
        self.linear_f = nn.Linear(embed_size * 2 , embed_size)

    def forward(self, images, lengths):
        """Extract image feature vectors."""
        #print("processing")
        x = images['nodes_flat']
        adj = images['adj_flat']
        batch = images['batch']
        #print(x)
        #print(adj)
        #print(batch)
        #adj = adj + adj.transpose(1, 2)
        adj = ((adj + adj.transpose(0, 1)) > 0.).float()
        edge_index, _ = dense_to_sparse(adj)

        x = self.embed(x)
        #x = x.squeeze(0)
        x = F.relu(self.gnn_embed(x, edge_index))
        xs = [global_mean_pool(x, batch)]
        x, edge_index, _, batch, _, _ =self.gnn_pool(x, edge_index, batch=batch)


        x = self.gnn_embed_f(x, edge_index)

        xs.append(global_mean_pool(x, batch))
        feature_out = self.linear_f(torch.cat(xs, -1))
        #feature_out = global_mean_pool(x, batch)
        if not self.no_imgnorm:
            feature_out = l2norm(feature_out)

        # take the absolute value of the embedding (used in order embeddings)
        if self.use_abs:
            feature_out = torch.abs(feature_out)


        spec_losses = torch.Tensor([0.])
        entr_losses = torch.Tensor([0.])
        coarsen_losses = torch.Tensor([0.])
        #feature_out = F.dropout(feature_out, 0.5)
        return feature_out, (spec_losses, entr_losses, coarsen_losses)



'''
class EncoderSSGPool_select(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, word_emb, use_abs=False, no_imgnorm=False,
                 use_specloss=False, use_entrloss=False, use_linkloss=False):
        """Load pretrained VGG19 and replace top fc layer."""
        super(EncoderSSGPool_select, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs
        self.use_specloss = use_specloss
        self.use_entrloss = use_entrloss
        self.use_linkloss = use_linkloss
        self.embed = word_emb
        self.num_layer = 2
        self.ratio = 0.5
        #self.att_linear = nn.Linear(embed_size, embed_size)
        #self.embed_final = nn.Linear(embed_size, embed_size)

        # Load a pre-trained model
        self.gnn_embed = DenseGCNConv(word_dim, embed_size) #GNN_Block(word_dim, embed_size, embed_size)##GNN_Block(word_dim, embed_size, embed_size)
        #self.gnn_pool =DenseGCNConv(embed_size, 1) #GNN_Block(embed_size, embed_size, 1)#
        self.gnn_pool_list = torch.nn.ModuleList()
        self.gnn_embed_list = torch.nn.ModuleList()
        self.final_linear = nn.Linear(embed_size * self.num_layer, embed_size)
        self.final_linear_njump = nn.Linear(embed_size, embed_size)
        for i in range(self.num_layer - 1):
            #num_nodes = ceil(ratio * num_nodes)
            self.gnn_pool_list.append(DenseGCNConv(embed_size, 1)) #GNN_Block(embed_size, embed_size, 1)
            self.gnn_embed_list.append(DenseGCNConv(embed_size, embed_size)) #GNN_Block(word_dim, embed_size, embed_size)
        #self.gnn_embed2 = GNN_Block(word_dim, embed_size, embed_size)#DenseGCNConv(embed_size, embed_size)
        #self.gnn_pool2 = GNN_Block(embed_size, embed_size, 1)
        #self.gnn3 = DenseGCNConv(embed_size, embed_size)

    def forward(self, images, lengths):
        """Extract image feature vectors."""
        x = images['nodes']
        adj = images['adj']
        adj = ((adj + adj.transpose(1, 2))>0.).float()
        spec_losses = 0.
        entr_losses = 0.
        link_losses = 0.
        B, N, _ = adj.size()
        mask = torch.arange(max(lengths)).expand(len(lengths), max(lengths)).cuda() < lengths.unsqueeze(1)
        mask_ori = mask
        P_final = torch.eye(N).unsqueeze(0).expand(B, N, N).cuda()
        P_inv_final = torch.eye(N).unsqueeze(0).expand(B, N, N).cuda()

        x = self.embed(x)
        x = self.gnn_embed(x, adj, mask)
        #xs = [x.sum(dim=1) / (mask.sum(-1).unsqueeze(-1) + 1e-10)]

        diag_ele = torch.sum(adj, -1)  # , keepdim=True)
        Diag = torch.diag_embed(diag_ele)
        lapl = Diag - adj
        lapl_ori = lapl
        adj_ori = adj

        for i, (gnn_pool, gnn_embed) in enumerate(
                zip(self.gnn_pool_list, self.gnn_embed_list)):
            x = F.relu(x)
            a = gnn_pool(x, adj, mask)
            x_next, adj_next, mask_next, lapl_next, P, P_inv = dense_ssg_pool(x, lapl, a, mask, ratio=self.ratio)
            x = gnn_embed(x_next, adj_next, mask_next)

            adj = adj_next
            lapl = lapl_next
            mask = mask_next

            #xs.append(x.sum(dim=1) / (mask.sum(-1).unsqueeze(-1) + 1e-10))

            P_final = torch.bmm(P, P_final)
            #P_inv_final = torch.bmm(P_inv_final, P_inv)

        P_inv_final = P_final.transpose(1, 2) / ((P_final * P_final).sum(dim=-1).unsqueeze(1) + 1e-10)
        #features = F.dropout(features, 0.5)
        #feature_out = self.final_linear(torch.cat(xs, dim=-1))
        #feature_out = self.final_linear_njump(x)
        feature_out = x.sum(dim=1) / (mask.sum(-1).unsqueeze(-1) + 1e-10)

        spec_loss = get_Spectral_loss_lap(lapl_ori, lapl, P_inv_final, mask_ori)
        #spec_loss = adj_ori - torch.matmul(P_final.transpose(1, 2), P_final)

        #identity = torch.eye(N).unsqueeze(0).expand(B, N, N).cuda()
        #adj_ori = adj_ori + identity# for new link loss
        link_loss = adj_ori - torch.matmul(P_final.transpose(1, 2), P_final)
        mask_ = mask_ori.view(B, N, 1).to(x.dtype)
        link_loss = link_loss * mask_
        link_loss = link_loss * mask_.transpose(1, 2)
        link_loss = torch.sqrt((link_loss * link_loss).sum(dim=(1, 2)))

        entr_loss = (-P_final * torch.log(P_final + 1e-10)).sum(dim=1)
        entr_loss = entr_loss.sum(-1) / (mask_ori.sum(-1) + 1e-10)

        spec_losses += spec_loss.mean()
        entr_losses += entr_loss.mean()
        link_losses += link_loss.mean()


        if not self.no_imgnorm:
            feature_out = l2norm(feature_out)

        # take the absolute value of the embedding (used in order embeddings)
        if self.use_abs:
            feature_out = torch.abs(feature_out)
        #feature_out = F.dropout(feature_out, 0.5)
        return feature_out, (spec_losses, entr_losses, link_losses)

    def forward_vis(self, images, lengths):
        """Extract image feature vectors."""
        x = images['nodes']
        adj = images['adj']
        adj = adj + adj.transpose(1, 2)

        B, N, _ = adj.size()
        mask = torch.arange(max(lengths)).expand(len(lengths), max(lengths)).cuda() < lengths.unsqueeze(1)
        mask_ori = mask
        P_final = torch.eye(N).unsqueeze(0).expand(B, N, N).cuda()
        P_inv_final = torch.eye(N).unsqueeze(0).expand(B, N, N).cuda()

        x = self.embed(x)
        x = self.gnn_embed(x, adj, mask)
        #xs = [x.sum(dim=1) / (mask.sum(-1).unsqueeze(-1) + 1e-10)]

        diag_ele = torch.sum(adj, -1)  # , keepdim=True)
        Diag = torch.diag_embed(diag_ele)
        lapl = Diag - adj
        lapl_ori = lapl
        adj_ori = adj

        for i, (gnn_pool, gnn_embed) in enumerate(
                zip(self.gnn_pool_list, self.gnn_embed_list)):
            x = F.relu(x)
            a = gnn_pool(x, adj, mask)
            x_next, adj_next, mask_next, lapl_next, P, P_inv = dense_ssg_pool(x, lapl, a, mask, ratio=self.ratio)
            x = gnn_embed(x_next, adj_next, mask_next)

            adj = adj_next
            lapl = lapl_next
            mask = mask_next

            #xs.append(x.sum(dim=1) / (mask.sum(-1).unsqueeze(-1) + 1e-10))

            P_final = torch.bmm(P, P_final)
            #P_inv_final = torch.bmm(P_inv_final, P_inv)

        P_inv_final = P_final.transpose(1, 2) / ((P_final * P_final).sum(dim=-1).unsqueeze(1) + 1e-10)
        #features = F.dropout(features, 0.5)
        #feature_out = self.final_linear(torch.cat(xs, dim=-1))
        #feature_out = self.final_linear_njump(x)
        feature_out = x.sum(dim=1) / (mask.sum(-1).unsqueeze(-1) + 1e-10)



        if not self.no_imgnorm:
            feature_out = l2norm(feature_out)

        # take the absolute value of the embedding (used in order embeddings)
        if self.use_abs:
            feature_out = torch.abs(feature_out)

        #feature_out = F.dropout(feature_out, 0.5)
        return feature_out, P_final
'''




# tutorials/09 - Image Captioning


class EncoderImageFull(nn.Module):

    def __init__(self, embed_size, finetune=False, cnn_type='vgg19',
                 use_abs=False, no_imgnorm=False):
        """Load pretrained VGG19 and replace top fc layer."""
        super(EncoderImageFull, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs

        # Load a pre-trained model
        self.cnn = self.get_cnn(cnn_type, True)

        # For efficient memory usage.
        for param in self.cnn.parameters():
            param.requires_grad = finetune

        # Replace the last fully connected layer of CNN with a new one
        if cnn_type.startswith('vgg'):
            self.fc = nn.Linear(self.cnn.classifier._modules['6'].in_features,
                                embed_size)
            self.cnn.classifier = nn.Sequential(
                *list(self.cnn.classifier.children())[:-1])
        elif cnn_type.startswith('resnet'):
            self.fc = nn.Linear(self.cnn.module.fc.in_features, embed_size)
            self.cnn.module.fc = nn.Sequential()

        self.init_weights()

    def get_cnn(self, arch, pretrained):
        """Load a pretrained CNN and parallelize over GPUs
        """
        if pretrained:
            print("=> using pre-trained model '{}'".format(arch))
            model = models.__dict__[arch](pretrained=True)
        else:
            print("=> creating model '{}'".format(arch))
            model = models.__dict__[arch]()

        if arch.startswith('alexnet') or arch.startswith('vgg'):
            model.features = nn.DataParallel(model.features)
            model.cuda()
        else:
            model = nn.DataParallel(model).cuda()

        return model

    def load_state_dict(self, state_dict):
        """
        Handle the models saved before commit pytorch/vision@989d52a
        """
        if 'cnn.classifier.1.weight' in state_dict:
            state_dict['cnn.classifier.0.weight'] = state_dict[
                'cnn.classifier.1.weight']
            del state_dict['cnn.classifier.1.weight']
            state_dict['cnn.classifier.0.bias'] = state_dict[
                'cnn.classifier.1.bias']
            del state_dict['cnn.classifier.1.bias']
            state_dict['cnn.classifier.3.weight'] = state_dict[
                'cnn.classifier.4.weight']
            del state_dict['cnn.classifier.4.weight']
            state_dict['cnn.classifier.3.bias'] = state_dict[
                'cnn.classifier.4.bias']
            del state_dict['cnn.classifier.4.bias']

        super(EncoderImageFull, self).load_state_dict(state_dict)

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        features = self.cnn(images)

        # normalization in the image embedding space
        features = l2norm(features)

        # linear projection to the joint embedding space
        features = self.fc(features)

        # normalization in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features)

        # take the absolute value of the embedding (used in order embeddings)
        if self.use_abs:
            features = torch.abs(features)

        regloss = (torch.Tensor([0.]), torch.Tensor([0.]), torch.Tensor([0.]))  # 0.
        return features, regloss




class EncoderImagePrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, use_abs=False, no_imgnorm=False):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs

        self.fc = nn.Linear(img_dim, embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features)

        # take the absolute value of embedding (used in order embeddings)
        if self.use_abs:
            features = torch.abs(features)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecomp, self).load_state_dict(new_state)


# tutorials/08 - Language Model
# RNN Based Language Model
class EncoderText(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers, word_emb,
                 use_abs=False):
        super(EncoderText, self).__init__()
        self.use_abs = use_abs
        self.embed_size = embed_size

        # word embedding
        #self.embed = nn.Embedding(vocab_size, word_dim)
        self.embed = word_emb
        # caption embedding
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True)

        #self.init_weights()

    #def init_weights(self):
    #    self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, ht = self.rnn(packed)

        # this code does not work
        # Reshape *final* output to (batch_size, hidden_size)
        # padded = pad_packed_sequence(out, batch_first=True)
        # I = torch.LongTensor(lengths).view(-1, 1, 1)
        # I = Variable(I.expand(x.size(0), 1, self.embed_size)-1).cuda()
        # out = torch.gather(padded[0], 1, I).squeeze(1)
        out = ht[-1]

        # normalization in the joint embedding space
        out = l2norm(out)

        # take absolute value, used by order embeddings
        if self.use_abs:
            out = torch.abs(out)

        return out


class EncoderSbert(nn.Module):
    def __init__(self,  embed_size, sbert_dim=1024, bert_name='bert-large-nli-stsb-mean-tokens',
                 use_abs=False, norm=False, finetune=False):
        super(EncoderSbert, self).__init__()
        self.use_abs = use_abs
        self.embed_size = embed_size
        self.embed = nn.Linear(sbert_dim, embed_size)
        self.sbert = SentenceTransformer(bert_name)
        self.norm = norm
        self.finetune = finetune
        print(f'Using BERT: {bert_name}')
        print('SBERT is being fine-tuned')

    def forward(self, x, *args, **kwargs):
        x = self.sbert.encode(x, batch_size=8, show_progress_bar=False, train=self.finetune)
        if not self.finetune:
            x = torch.stack([torch.tensor(xx, dtype=torch.float) for xx in x])
            #x = F.dropout(x, 0.1)
            out = self.embed(x.cuda())
        else:
            out = self.embed(torch.stack(x))

        #out = F.normalize(out, p=2, dim=-1)
        if self.norm:
            out = l2norm(out)

        if self.use_abs:
            out = torch.abs(out)
        #out = F.dropout(out,0.5)
        return out


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        if measure == 'order':
            self.sim = order_sim
        else:
            self.sim = cosine_sim

        self.max_violation = max_violation

    def forward(self, im, s):
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.mean() + cost_im.mean()#cost_s.sum() + cost_im.sum()


class VSE(object):
    """
    rkiros/uvs model
    """

    def __init__(self, data_name='coco', img_dim=4096, finetune=False, cnn_type='vgg16', gmodel_type=None,
                 use_abs=False, no_imgnorm=False, no_txtnorm=False, word_dim=300,word_dim_sg=300, embed_size=1024, grad_clip=2,
                 learning_rate=0.0002, margin=0.2, sbert=False, max_violation=False, vocab_size=11755, sg_vocab_size=0,
                 word_emb=None, measure='cosine', finetune_bert=False, use_SG=False, use_specloss=False, use_entrloss=False, use_linkloss=False, lambda_reg = 1.0, num_layers = 1, pool_ratio=0.1,  **kwargs):
        # tutorials/09 - Image Captioningƒ
        # Build Models
        self.grad_clip = grad_clip
        self.use_SG = use_SG
        self.use_specloss = use_specloss
        self.use_entrloss = use_entrloss
        self.use_linkloss = use_linkloss
        self.lambda_reg = lambda_reg
        if use_SG:
            self.img_word_emb = Encoder_word(sg_vocab_size, word_dim_sg, word_emb)
        else:
            self.img_word_emb = None
        self.txt_word_emb = Encoder_word(vocab_size, word_dim)
        self.img_enc = EncoderImage(data_name, img_dim, sg_vocab_size, word_dim_sg, embed_size, self.img_word_emb,
                                    finetune, cnn_type, gmodel_type,
                                    use_abs=use_abs,
                                    no_imgnorm=no_imgnorm,
                                    use_SG=use_SG,
                                    use_specloss=use_specloss, use_entrloss=use_entrloss, use_linkloss=use_linkloss,
                                    num_layers=num_layers, pool_ratio=pool_ratio)
        if sbert:
            self.txt_enc = EncoderSbert(embed_size,
                                        use_abs=use_abs,
                                        norm=not no_txtnorm,
                                        finetune=finetune_bert,
                                        sbert_dim=word_dim,
                                        bert_name=sbert)

        else:
            self.txt_enc = EncoderText(vocab_size, word_dim,
                                       embed_size, num_layers, self.txt_word_emb,
                                       use_abs=use_abs)
        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = ContrastiveLoss(margin=margin,
                                         measure=measure,
                                         max_violation=max_violation)

        params = list(self.txt_word_emb.parameters())
        params += list(self.txt_enc.parameters())
        if use_SG:
            #params += list(self.img_word_emb.parameters())
            params += list(self.img_enc.parameters())
        else:
            params += list(self.img_enc.fc.parameters())
            if finetune:
                params += list(self.img_enc.cnn.parameters())
        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=learning_rate)

        self.Eiters = 0


    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()
    def infer_img_feat(self, images, img_lengths):
        """Compute features of the image
        """
        if torch.cuda.is_available():
            if self.use_SG == True:
                images['nodes'] = images['nodes'].cuda()
                images['adj'] = images['adj'].cuda()
                images['nodes_flat'] = images['nodes_flat'].cuda()
                images['adj_flat'] = images['adj_flat'].cuda()
                images['batch'] = images['batch'].cuda()
                img_lengths = img_lengths.cuda()
            else:
                images = images.cuda()
        with torch.no_grad():
            if self.use_SG == True:
                img_emb, _ = self.img_enc(images, img_lengths)
            else:
                img_emb, _ = self.img_enc(images)

        return img_emb

    def infer_txt_feat(self, captions, lengths):
        """Compute features of the caption
        """
        captions = captions.unsqueeze(0) if captions.dim() == 1 else captions
        if torch.cuda.is_available():
            if not isinstance(captions, list):
                captions = captions.cuda()
        with torch.no_grad():
            cap_emb = self.txt_enc(captions, lengths)

        return cap_emb

    def infer_txt2img_retrieval(self, cap_embs, img_embs):
        if type(img_embs) is np.ndarray:
            img_embs = torch.from_numpy(img_embs).float()
        cap_embs = cap_embs.unsqueeze(0) if cap_embs.dim() == 1 else cap_embs
        if torch.cuda.is_available():
            cap_embs = cap_embs.cuda()
            img_embs = img_embs.cuda()
        d = torch.bmm(cap_embs, img_embs.transpose(0, 1))

        inds = torch.argsort(d)

        return inds


    def forward_vis(self, images, captions, lengths, img_lengths):
        """Compute the image and caption for visualizations
        """

        if torch.cuda.is_available():
            if self.use_SG == True:
                images['nodes'] = images['nodes'].cuda()
                images['adj'] = images['adj'].cuda()
                images['nodes_flat'] = images['nodes_flat'].cuda()
                images['adj_flat'] = images['adj_flat'].cuda()
                images['batch'] = images['batch'].cuda()
                img_lengths = img_lengths.cuda()
            else:
                images = images.cuda()
            if not isinstance(captions, list):
                captions = captions.cuda()


        with torch.no_grad():
            if self.use_SG == True:
                img_emb, P_final = self.img_enc.forward_vis(images, img_lengths)
            else:
                img_emb, regloss = self.img_enc(images)
            cap_emb = self.txt_enc(captions, lengths)


        return img_emb, cap_emb, P_final

    def forward_emb(self, images, captions, lengths, img_lengths, volatile=False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        if torch.cuda.is_available():
            if self.use_SG == True:
                images['nodes'] = images['nodes'].cuda()
                images['adj'] = images['adj'].cuda()
                images['nodes_flat'] = images['nodes_flat'].cuda()
                images['adj_flat'] = images['adj_flat'].cuda()
                images['batch'] = images['batch'].cuda()
                img_lengths = img_lengths.cuda()
            else:
                images = images.cuda()
            if not isinstance(captions, list):
                captions = captions.cuda()

        # Forward
        if volatile:
            with torch.no_grad():
                if self.use_SG == True:
                    img_emb, regloss = self.img_enc(images, img_lengths)
                else:
                    img_emb, regloss = self.img_enc(images)
                cap_emb = self.txt_enc(captions, lengths)
        else:
            if self.use_SG == True:
                img_emb, regloss = self.img_enc(images, img_lengths)
            else:
                img_emb, regloss = self.img_enc(images)
            cap_emb = self.txt_enc(captions, lengths)



        return img_emb, cap_emb, regloss

    def forward_loss(self, img_emb, cap_emb, regloss, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(img_emb, cap_emb)
        self.logger.update('Le', loss.item(), img_emb.size(0))
        for i, reg in enumerate(regloss):
            if i==0:
                self.logger.update('Spec_loss', reg.item(), img_emb.size(0))
                if self.use_specloss !=False:
                    loss += self.lambda_reg * reg
            if i==1:
                self.logger.update('Entr_loss', reg.item(), img_emb.size(0))
                if self.use_entrloss !=False:
                    loss += self.lambda_reg * reg
            if i==2:
                self.logger.update('Link_loss', reg.item(), img_emb.size(0))
                if self.use_linkloss !=False:
                    loss += self.lambda_reg * reg
            #loss += 10. * reg

        #loss += regloss
        # self.logger.update('Le', loss.data[0], img_emb.size(0))

        return loss

    def train_emb(self, images, captions, lengths, img_lengths, ids=None, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_emb, cap_emb, regloss = self.forward_emb(images, captions, lengths, img_lengths)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(img_emb, cap_emb, regloss)

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()

        #del loss, regloss

