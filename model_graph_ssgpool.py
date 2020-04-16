from math import ceil

import torch
import time
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import DenseSAGEConv, DenseGCNConv,  JumpingKnowledge


EPS = 1e-7
def arccosh(x):
    x = x + EPS
    #c0 = torch.log(x)
    ##print(c0)
    #c1 = torch.log1p(torch.sqrt(x * x - 1) / x)
    #return c0 + c1
    a = torch.sqrt(x*x-1)
    return torch.log(x+a)

def dense_diff_pool(x, adj, s, mask=None):
    r"""Differentiable pooling operator from the `"Hierarchical Graph
    Representation Learning with Differentiable Pooling"
    <https://arxiv.org/abs/1806.08804>`_ paper

    .. math::
        \mathbf{X}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{X}

        \mathbf{A}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{A} \cdot \mathrm{softmax}(\mathbf{S})

    based on dense learned assignments :math:`\mathbf{S} \in \mathbb{R}^{B
    \times N \times C}`.
    Returns pooled node feature matrix, coarsened adjacency matrix and two
    auxiliary objectives: (1) The link prediction loss

    .. math::
        \mathcal{L}_{LP} = {\| \mathbf{A} -
        \mathrm{softmax}(\mathbf{S}) {\mathrm{softmax}(\mathbf{S})}^{\top}
        \|}_F,

    and the entropy regularization

    .. math::
        \mathcal{L}_E = \frac{1}{N} \sum_{n=1}^N H(\mathbf{S}_n).

    Args:
        x (Tensor): Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B
            \times N \times F}` with batch-size :math:`B`, (maximum)
            number of nodes :math:`N` for each graph, and feature dimension
            :math:`F`.
        adj (Tensor): Adjacency tensor :math:`\mathbf{A} \in \mathbb{R}^{B
            \times N \times N}`.
        s (Tensor): Assignment tensor :math:`\mathbf{S} \in \mathbb{R}^{B
            \times N \times C}` with number of clusters :math:`C`. The softmax
            does not have to be applied beforehand, since it is executed
            within this method.
        mask (BoolTensor, optional): Mask matrix
            :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
            the valid nodes for each graph. (default: :obj:`None`)

    :rtype: (:class:`Tensor`, :class:`Tensor`, :class:`Tensor`,
        :class:`Tensor`)
    """

    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    batch_size, num_nodes, _ = x.size()

    s = torch.softmax(s, dim=-1)

    if mask is not None:
        mask_ = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask_, s * mask_

    out = torch.matmul(s.transpose(1, 2), x)
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

    #identity = torch.eye(num_nodes).unsqueeze(0).expand(batch_size, num_nodes, num_nodes).cuda()
    #link_loss = (adj+identity) - torch.matmul(s, s.transpose(1, 2))
    link_loss = adj - torch.matmul(s, s.transpose(1, 2))
    link_loss = link_loss * mask_
    link_loss = link_loss * mask_.transpose(1, 2)

    link_loss = torch.sqrt((link_loss*link_loss).sum(dim=(1, 2)))#torch.norm(link_loss, p=2, dim=-1)
    #link_loss = link_loss.sum(-1) / (mask.sum(-1) + 1e-10)
    #link_loss = link_loss.sum()  # / batch_size # / adj.numel()



    ent_loss = (-s * torch.log(s + EPS)).sum(dim=-1)  # .mean()
    ent_loss = ent_loss.sum(-1) / (mask.sum(-1) + 1e-10)
    #ent_loss = ent_loss.sum()

    #spec_loss = get_Spectral_loss(adj, out_adj, s, mask)


    return out, out_adj, link_loss.mean(), ent_loss.mean(), torch.Tensor([0.])



def dense_ssgpool(x, adj, s, mask=None, debug=False):


    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    batch_size, num_nodes, _ = x.size()

    s = torch.softmax(s, dim=-1)

    if mask is not None:
        mask_ = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask_, s * mask_

    diag_ele = torch.sum(adj, -1)
    Diag = torch.diag_embed(diag_ele)
    Lapl = Diag - adj
    L_next = torch.matmul(torch.matmul(s.transpose(1, 2), Lapl), s)

    identity = torch.eye(s.size(-1)).unsqueeze(0).expand(batch_size, s.size(-1), s.size(-1)).cuda()
    padding = torch.zeros_like(identity)

    A_next = -torch.where(torch.eq(identity, 1.), padding, L_next)

    '''Get feature matrix of next layer'''

    s_inv = s.transpose(1, 2) / ((s * s).sum(dim=1).unsqueeze(
        -1) + EPS)  # Pseudo-inverse of P (easy inversion theorem, in this case, sum is same with 2-norm)

    #s_inv_T = P_inv.transpose(1, 2)

    x_next = torch.bmm(s_inv, x)

    #identity = torch.eye(num_nodes).unsqueeze(0).expand(batch_size, num_nodes, num_nodes).cuda()
    #link_loss = (adj + identity) - torch.matmul(s, s.transpose(1, 2))

    return x_next, A_next, Lapl, L_next, s, s_inv


"""
def dense_ssg_pool(x, lapl, a, mask=None, ratio=0.5, is_training=False, debug=False):
    '''
    Args:
        x (Tensor): Node feature tensor :[B, N, dim]
        adj (Tensor): Adjacency tensor :[B, N, N]
        a (Tensor): score tensor :math:`[B, N, 1]
        ratio (Float): graph pooling ratio
        num_nodes (long? Tensor): numober of nodes tensor: [B]
        mask (BoolTensor, optional): Mask matrix
            :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
            the valid nodes for each graph. (default: :obj:`None`)

    '''

    x = x.unsqueeze(0) if x.dim() == 2 else x
    lapl = lapl.unsqueeze(0) if lapl.dim() == 2 else lapl
    a = a.unsqueeze(0) if a.dim() == 2 else a
    B, N, dim = x.size()
    num_nodes = mask.sum(-1)

    if mask is not None:
        mask_ = mask.view(B, N, 1).to(x.dtype)
        x = x * mask_
        pad = torch.ones_like(mask_) * (-1e5)
        a = torch.where(torch.eq(mask_, 0.), pad, a)
        lapl = lapl * mask_
        lapl = lapl * mask_.transpose(1, 2)

    next_num_nodes = torch.ceil(ratio * num_nodes)

    '''Get Coarsening matrix P'''
    P, next_mask = get_coarsen_matrix_select(x, a, mask, next_num_nodes)

    '''Get Laplacian matrix of next layer'''
    #diag_ele = torch.sum(adj, -1)  # , keepdim=True)
    #Diag = torch.diag_embed(diag_ele)
    #Lapl = Diag - adj
    PL = torch.bmm(P, lapl)
    L_next = torch.bmm(PL, P.transpose(1, 2))


    '''Get Adjacency matrix of next layer'''
    identity = torch.eye(N).unsqueeze(0).expand(B, N, N).cuda()
    padding = torch.zeros_like(identity)

    A_next = -torch.where(torch.eq(identity, 1.), padding, L_next)

    '''Get feature matrix of next layer'''
    P_inv = P.transpose(1, 2) / ((P * P).sum(dim=-1).unsqueeze(
        1) + 1e-10)  # Pseudo-inverse of P (easy inversion theorem, in this case, sum is same with 2-norm)
    P_inv_T = P_inv.transpose(1, 2)
    x_next = torch.bmm(P_inv_T, x)

    return x_next, A_next, next_mask, L_next, P, P_inv
"""
'''
def get_coarsen_matrix_select(x, a, mask, next_num_nodes, is_training=False):
    """

    :param a:
    :param adj_i:
    :param next_num_nodes:
    :return:
    """

    B, N, dim = x.size()
    next_num_nodes = next_num_nodes.unsqueeze(1).expand(B, N)
    seq_range = torch.arange(N).unsqueeze(0).expand(B, N).cuda()
    next_mask = seq_range < next_num_nodes


    sorted_a, indices_a = torch.sort(a.squeeze(-1), dim=1, descending=True)


    x_idx = indices_a.unsqueeze(-1).expand(B, N, dim)
    sorted_x = torch.gather(x, 1, x_idx)


    P = torch.bmm(sorted_x, x.transpose(1,2))


    pad = torch.ones_like(P) * (-float("inf"))
    next_mask_ = next_mask.unsqueeze(-1).expand(B, N, N)
    P = torch.where(torch.eq(next_mask_, False), pad, P)


    P = torch.softmax(P, dim=1) #gumbel_softmax(P, axis=1, hard=False, is_training=is_training)
    nanchecker = torch.isnan(P)
    P = torch.where(nanchecker, torch.zeros_like(P), P)

    pad = torch.zeros_like(P)
    mask_ = mask.unsqueeze(1).expand(B, N, N)
    P = torch.where(torch.eq(mask_, False), pad, P)

    return P, next_mask
'''

'''
def get_Spectral_loss(adj, new_adj, s, mask=None, debug=False):
    """

    :param L_1:
    :param L_2:
    :return:
    """
    #print("GET SPECTRAL LOSS")
    #P_inv.register_hook(print)

    B, N, _ = adj.size()
    #TODO: detach?


    diag_ele = torch.sum(adj, -1)  # , keepdim=True)
    Diag = torch.diag_embed(diag_ele)
    L_1 = Diag - adj

    s_inv = s.transpose(1, 2) / ((s * s).sum(dim=1).unsqueeze(
        -1) + EPS)  # Pseudo-inverse of P (easy inversion theorem, in this case, sum is same with 2-norm)
    up_adj_2 = torch.bmm(s_inv.transpose(1, 2), new_adj)
    up_adj_2 = torch.bmm(up_adj_2, s_inv)

    identity = torch.eye(N).unsqueeze(0).expand(B, N, N).cuda()

    if mask is not None:
        mask_ = mask.unsqueeze(-1)
        up_adj_2 = up_adj_2 * mask_
        up_adj_2 = up_adj_2 * mask_.transpose(1, 2)
        identity = identity * mask_
        identity = identity * mask_.transpose(1, 2)

    diag_ele = torch.sum(up_adj_2, -1)  # , keepdim=True)
    Diag = torch.diag_embed(diag_ele)
    up_L_2 = Diag - up_adj_2

    up_L_2 = up_L_2 + identity * 1e-4

    e, v = torch.symeig(L_1.cpu(), eigenvectors=True)
    e = e.cuda()#.detach()
    v = v.cuda()#.detach()


    e[e < 1e-4] = 0.  # ignore small value (to zero eivenvalue)


    fiedler_idx = (e==0.).sum(dim=1)
    fiedler_idx = fiedler_idx.unsqueeze(-1).expand(B, N).unsqueeze(-1)

    fv = torch.gather(v, dim=2, index=fiedler_idx)#.detach()

    term_1 = (torch.bmm((L_1 - up_L_2), fv) * torch.bmm((L_1 - up_L_2), fv)).squeeze(-1).sum(-1)
    term_2 = (fv * fv).squeeze(-1).sum(-1)
    fv_L1 = torch.bmm(fv.transpose(1, 2), L_1)
    term_3 = torch.bmm(fv_L1, fv).squeeze(-1).squeeze(-1)
    fv_L2 = torch.bmm(fv.transpose(1, 2), up_L_2)
    term_4 = torch.bmm(fv_L2, fv).squeeze(-1).squeeze(-1)

    spectral_loss = arccosh(1 + (term_1 * term_2)/(2 * term_3 * term_4 + EPS))

    nanchecker = torch.isnan(spectral_loss)
    spectral_loss = torch.where(nanchecker, torch.zeros_like(spectral_loss), spectral_loss)
    return spectral_loss#torch.mean(spectral_loss)
'''
def get_Spectral_loss(L_1, L_2, s_inv, num_loss=1, mask=None, debug=False):
    """

    :param L_1:
    :param L_2:
    :return:
    """
    #print("GET SPECTRAL LOSS")
    #P_inv.register_hook(print)

    B, N, _ = L_1.size()
    #TODO: detach?


    up_L_2 = torch.bmm(s_inv, L_2)
    up_L_2 = torch.bmm(up_L_2, s_inv.transpose(1, 2))

    identity = torch.eye(N).unsqueeze(0).expand(B, N, N).cuda()
    if mask is not None:
        mask_ = mask.unsqueeze(-1)
        up_L_2 = up_L_2 * mask_
        up_L_2 = up_L_2 * mask_.transpose(1, 2)
        identity = identity * mask_
        identity = identity * mask_.transpose(1,2)



    e, v = torch.symeig(L_1.cpu(), eigenvectors=True)
    e = e.cuda()
    v = v.cuda()

    up_L_2 = up_L_2 + identity*1e-4
    #e_temp, v_temp = torch.symeig(L2_temp.cpu(),eigenvectors=True)

    e[e < 1e-4] = 0.  # ignore small value (to zero eivenvalue)

    fiedler_idx = (e==0.).sum(dim=1)

    fiedler_idx = fiedler_idx.unsqueeze(-1).expand(B, N).unsqueeze(-1)

    fv = torch.gather(v, dim=2, index=fiedler_idx)  # .detach()
    for i in range(num_loss-1):
        fv_1 = torch.gather(v, dim=2, index=fiedler_idx + i+1)  # .detach()
        fv = torch.cat((fv, fv_1), -1)

    term_1 = (torch.bmm((L_1 - up_L_2), fv) * torch.bmm((L_1 - up_L_2), fv)).sum(1)  # Bxnum_loss
    term_2 = (fv * fv).sum(1)  # Bxnum_loss
    fv_L1 = torch.bmm(fv.transpose(1, 2), L_1)
    term_3 = torch.diagonal(torch.bmm(fv_L1, fv), dim1=1, dim2=2)  # Bxnum_loss
    fv_L2 = torch.bmm(fv.transpose(1, 2), up_L_2)
    term_4 = torch.diagonal(torch.bmm(fv_L2, fv), dim1=1, dim2=2)  # Bxnum_loss

    term_mask = torch.zeros_like(term_1)
    term_1 = torch.where(term_1 < 0., term_mask, term_1)
    term_2 = torch.where(term_2 < 0., term_mask, term_2)
    term_3 = torch.where(term_3 < 0., term_mask, term_3)
    term_4 = torch.where(term_4 < 0., term_mask, term_4)


    spectral_loss = arccosh(1 + (term_1 * term_2) / (2 * term_3 * term_4 + EPS))
    spectral_loss = spectral_loss.mean(-1)
    #spectral_loss = arccosh(1 + (term_1 * term_2) / (2 * term_3  + EPS))

    #nanchecker = torch.isnan(spectral_loss)
    #spectral_loss = torch.where(nanchecker, torch.zeros_like(spectral_loss), spectral_loss )
    return spectral_loss



