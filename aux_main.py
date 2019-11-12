import argparse
import torch
import os
import numpy as np
import torch.optim as optim
from preprocess.utils import load_files, save_json
from dataloader import Dset_VG, Dset_VG_inference, Dset_VG_inference_AUX, Dset_VG_Pairwise, Dset_VG_Pairwise_AUX, get_word_vec, get_sim, he_sampling
from torch.utils.data import DataLoader
from model import HGAN, HGAN_AUX, HGAN_LATE_V1
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from glob import glob
import pandas as pd


def prop_mini_batch_infer(mini_batch_compare, vg_id_anchor, HE_anchor, label, label_id2idx, model):
    HE_compare, vg_id_compare = mini_batch_compare
    sim_score = [get_sim(vg_id_anchor, i, label, label_id2idx) for i in vg_id_compare]

    HE_compare = HE_compare.cuda()
    HE_anchor = torch.from_numpy(HE_anchor).cuda()
    sim_score = torch.reshape(torch.tensor(sim_score), (-1, 1)).cuda()

    num_batch = HE_compare.shape[0]
    HE_anchor = HE_anchor.unsqueeze(0)
    batch_anchor = HE_anchor.repeat([num_batch, 1, 1])

    if model.cfg['MODEL']['TARGET'] == 'SBERT':
        score, loss, att_map = model(batch_anchor, HE_compare, sim_score, 'infer')
    else:
        score, _, loss = model(batch_anchor, HE_compare, HE_compare)
    return score, att_map


def prop_mini_batch_infer_aux(mini_batch_compare, vg_id_anchor, HE_anchor, aux_anchor, label, label_id2idx, model):
    HE_compare, vg_id_compare, aux_compare = mini_batch_compare
    sim_score = [get_sim(vg_id_anchor, i, label, label_id2idx) for i in vg_id_compare]

    HE_compare = HE_compare.cuda()
    # HE_anchor = torch.from_numpy(HE_anchor).cuda()
    HE_anchor = (HE_anchor).cuda()
    sim_score = torch.reshape(torch.tensor(sim_score), (-1, 1)).cuda()

    num_batch = HE_compare.shape[0]
    HE_anchor = HE_anchor.unsqueeze(0)
    batch_anchor = HE_anchor.repeat([num_batch, 1, 1])

    aux_anchor = aux_anchor.unsqueeze(0)
    batch_aux_anchor = aux_anchor.repeat([num_batch, 1])
    batch_aux_anchor = batch_aux_anchor.cuda()
    aux_compare = aux_compare.cuda()

    if model.cfg['MODEL']['TARGET'] == 'SBERT':
        score, loss, att_map = model(batch_anchor, HE_compare, sim_score, aux_anchor=batch_aux_anchor, aux_pos=aux_compare, mode='infer')
    else:
        score, _, loss = model(batch_anchor, HE_compare, HE_compare)
    return score, att_map



def prop_mini_batch_sbert(mini_batch, model):
    HE_a, HE_p, sbert_score = mini_batch
    HE_a = HE_a.cuda()
    if len(HE_a.shape) == 4:
        num_batch = HE_a.shape[0]
        num_samples = HE_a.shape[1]
        flat_HE_a = torch.reshape( HE_a, (num_batch*num_samples, HE_a.shape[2], HE_a.shape[3]) )
        HE_a = flat_HE_a.cuda()
        flat_HE_p = torch.reshape( HE_p, (num_batch*num_samples, HE_p.shape[2], HE_p.shape[3]) )
        flat_sbert_score = torch.reshape( sbert_score, (num_batch*num_samples, -1))
        HE_p = flat_HE_p.cuda()
        sbert_score = flat_sbert_score.cuda()
    else:
        HE_p = HE_p.cuda()
        sbert_score = sbert_score.cuda()

    score_p, loss = model(HE_a, HE_p, sbert_score)

    return score_p, loss


def prop_mini_batch_sbert_aux(mini_batch, model, resnet_weight=None):
    HE_a, HE_p, sbert_score, aux_anchor, aux_compare = mini_batch
    HE_a = HE_a.cuda()
    if len(HE_a.shape) == 4:
        num_batch = HE_a.shape[0]
        num_samples = HE_a.shape[1]
        flat_HE_a = torch.reshape( HE_a, (num_batch*num_samples, HE_a.shape[2], HE_a.shape[3]) )
        HE_a = flat_HE_a.cuda()
        flat_HE_p = torch.reshape( HE_p, (num_batch*num_samples, HE_p.shape[2], HE_p.shape[3]) )
        flat_sbert_score = torch.reshape( sbert_score, (num_batch*num_samples, -1))
        HE_p = flat_HE_p.cuda()
        sbert_score = flat_sbert_score.cuda()

        num_batch = aux_anchor.shape[0]
        num_samples = aux_anchor.shape[1]
        aux_anchor = aux_anchor.reshape(num_batch * num_samples, -1).cuda()
        aux_compare = aux_compare.reshape(num_batch * num_samples, -1).cuda()
        if resnet_weight is not None:
            dot = (aux_anchor * aux_compare).sum(dim=1)
            norm_anchor = aux_anchor.norm(dim=1)
            norm_compare = aux_compare.norm(dim=1)
            cosine = dot / norm_anchor / norm_compare + 1  # set similar scale to 0 ~ 2
            cosine = cosine.reshape(num_batch * num_samples, -1)
            sbert_score = sbert_score * (1 - resnet_weight) + cosine * (resnet_weight)
    else:
        HE_p = HE_p.cuda()
        sbert_score = sbert_score.cuda()
        raise NotImplementedError

    score_p, loss = model(HE_a, HE_p, sbert_score, aux_anchor, aux_compare)

    return score_p, loss



def prop_mini_batch(mini_batch, model):
    HE_a, HE_p, HE_n = mini_batch
    HE_a = HE_a.cuda()
    HE_p = HE_p.cuda()
    HE_n = HE_n.cuda()

    score_p, score_n, loss = model(HE_a, HE_p, HE_n)

    return score_p, score_n, loss


def inference(model, sg_test, infer_dset, infer_dloader, args):
    ckpt_path = args.ckpt_path + args.exp_name
    ckpts = glob(os.path.join(ckpt_path, 'ckpt_*.pth.tar'))

    if len(ckpts) == 0:
        print("Error!, No saved models!")
        return -1

    num = []
    for ckpt in ckpts:
        tokens = ckpt.split('.')
        num.append(int(tokens[-3].split('_')[-1]))

    num.sort()
    last_ckpt = os.path.join(ckpt_path, 'ckpt_' + str(num[-1]) + '.pth.tar')
    print("Load the last model, epoch: %d" % (num[-1]))

    checkpoint = torch.load(last_ckpt)
    model.load_state_dict(checkpoint['state_dict'])
    print("loaded checkpoint '{}' (epoch {})".format(ckpts[-1], checkpoint['idx_epoch']))

    result_path = args.result_path + args.exp_name

    result_viewer_path = '/data/project/rw/viewer_CBIR/viewer/results/{}_epoch_{}'.format(args.exp_name, num[-1])
    if not os.path.exists(result_viewer_path):
        os.makedirs(result_viewer_path)

    path_test_img_id_1000 = '/data/project/rw/CBIR/test_id_1000_v2.json'
    test_img_id_1000 = load_files(path_test_img_id_1000)

    while True:
        vids = input("visual genome image id, comma separated : ")
        if int(vids) == -1:
            break
        elif int(vids) == 100:
            vids = test_img_id_1000[:100]
        elif int(vids) == '1000':
            vids = test_img_id_1000
        else:
            vids = vids.strip().split(',')

        for vid in vids:
            vg_id_infer = str(vid.strip())
            infer_sg = sg_test[vg_id_infer]
            word_vec_anchor = get_word_vec(infer_sg, infer_dset.vocab2idx, infer_dset.vocab_glove)
            HE_anchor, HEs_anchor = he_sampling(infer_sg['adj'], word_vec_anchor, infer_dset.sampling_steps, infer_dset.max_num_he, 'infer')

            infer_result = {}
            for b_idx, mini_batch in enumerate(tqdm(infer_dloader)):
                # HE_compare, vg_id_compare = mini_batch
                score, att_map = prop_mini_batch_infer(mini_batch, vid, HE_anchor, infer_dset.label, infer_dset.label_id2idx, model)

                if args.visualize_att:
                    for idx_m in range(len(mini_batch[1])):
                        infer_result[mini_batch[1][idx_m]] = {"score": score[idx_m].item(), "att": att_map[idx_m].item()}
                else:
                    score_arr = [s.item() for s in score]
                    infer_result.update(list(zip(mini_batch[1], score_arr)))

            if args.visualize_att:
                save_json(infer_result, result_path + '/infer_result_w_att_epoch_{}_vid_{}.json'.format(num[-1], vid))
            else:
                data_pandas = pd.DataFrame.from_dict(infer_result, orient='index')
                data_pandas.to_csv(result_viewer_path+'/{}.tsv'.format(vid), sep='\t', header=False)#, index=False)
                save_json(infer_result, result_path + '/infer_result_w_att_epoch_{}_vid_{}.json'.format(num[-1], vid))


def inference_rerank(model, sg_test, infer_dset, args, model_cfg, label_similarity, label_vg_ids):
    ckpt_path = args.ckpt_path + args.exp_name
    ckpts = glob(os.path.join(ckpt_path, 'ckpt_*.pth.tar'))

    if len(ckpts) == 0:
        print("Error!, No saved models!")
        return -1

    if args.epoch is None:
        num = []
        for ckpt in ckpts:
            tokens = ckpt.split('.')
            num.append(int(tokens[-3].split('_')[-1]))

    else:
        num = [args.epoch]
    num.sort()
    last_ckpt = os.path.join(ckpt_path, 'ckpt_' + str(num[-1]) + '.pth.tar')
    print("Load the last model, epoch: %d" % (num[-1]))

    checkpoint = torch.load(last_ckpt)
    model.load_state_dict(checkpoint['state_dict'])
    print("loaded checkpoint '{}' (epoch {})".format(ckpts[-1], checkpoint['idx_epoch']))

    result_path = args.result_path + args.exp_name

    result_viewer_path = '/data/project/rw/viewer_CBIR/viewer/results/{}_epoch_{}_rerank'.format(args.exp_name, num[-1])
    if not os.path.exists(result_viewer_path):
        os.makedirs(result_viewer_path)

    path_test_img_id_1000 = '/data/project/rw/CBIR/test_id_1000_v3.json'
    test_img_id_1000 = load_files(path_test_img_id_1000)

    while True:
        vids = input("visual genome image id, comma separated : ")
        if vids.isnumeric():
            if int(vids) == -1:
                break
            elif int(vids) == 100:
                vids = test_img_id_1000[:100]
            elif int(vids) == 1000:
                vids = test_img_id_1000
        else:
            vids = vids.strip().split(',')

        for vid in tqdm(vids):
            vg_id_infer = str(vid.strip())
            infer_sg = sg_test[vg_id_infer]
            word_vec_anchor = get_word_vec(infer_sg, infer_dset.vocab2idx, infer_dset.vocab_glove)
            HE_anchor, HEs_anchor = he_sampling(infer_sg['adj'], word_vec_anchor, infer_dset.sampling_steps, infer_dset.max_num_he, 'infer')

            # RESNET RERANKING
            l_target_id = get_resnet_rank(vid)
            target_ids = set(l_target_id)
            sg_ranked = {k: v for k, v in sg_test.items() if k in target_ids}

            rerank_infer_dset = Dset_VG_inference(model_cfg, sg_ranked, label_similarity, label_vg_ids,
                                                  infer_dset.vocab_glove, infer_dset.vocab2idx)
            infer_dloader = DataLoader(rerank_infer_dset, batch_size=10, num_workers=10, shuffle=False)


            infer_result = {}
            for b_idx, mini_batch in enumerate((infer_dloader)):
                # HE_compare, vg_id_compare = mini_batch
                score, att_map = prop_mini_batch_infer(mini_batch, vid, HE_anchor, infer_dset.label, infer_dset.label_id2idx, model)

                if args.visualize_att:
                    for idx_m in range(len(mini_batch[1])):
                        infer_result[mini_batch[1][idx_m]] = {"score": score[idx_m].item(), "att": att_map[idx_m].item()}
                else:
                    score_arr = [s.item() for s in score]
                    infer_result.update(list(zip(mini_batch[1], score_arr)))

            if args.visualize_att:
                save_json(infer_result, result_path + '/infer_result_w_att_epoch_{}_vid_{}.json'.format(num[-1], vid))
            else:
                data_pandas = pd.DataFrame.from_dict(infer_result, orient='index')
                data_pandas.to_csv(result_viewer_path+'/{}.tsv'.format(vid), sep='\t', header=False)#, index=False)
                save_json(infer_result, result_path + '/infer_result_w_att_epoch_{}_vid_{}.json'.format(num[-1], vid))


def inference_rerank_AUX(model, sg_test, infer_dset, args, model_cfg, label_similarity, label_vg_ids):
    ckpt_path = args.ckpt_path + args.exp_name
    ckpts = glob(os.path.join(ckpt_path, 'ckpt_*.pth.tar'))

    if len(ckpts) == 0:
        print("Error!, No saved models!")
        return -1

    num = []
    for ckpt in ckpts:
        tokens = ckpt.split('.')
        num.append(int(tokens[-3].split('_')[-1]))

    num.sort()
    last_ckpt = os.path.join(ckpt_path, 'ckpt_' + str(num[-1]) + '.pth.tar')
    print("Load the last model, epoch: %d" % (num[-1]))

    checkpoint = torch.load(last_ckpt)
    model.load_state_dict(checkpoint['state_dict'])
    print("loaded checkpoint '{}' (epoch {})".format(ckpts[-1], checkpoint['idx_epoch']))

    result_path = args.result_path + args.exp_name

    result_viewer_path = '/data/project/rw/viewer_CBIR/viewer/results/{}_epoch_{}_rerank'.format(args.exp_name, num[-1])
    if not os.path.exists(result_viewer_path):
        os.makedirs(result_viewer_path)

    path_test_img_id_1000 = '/data/project/rw/CBIR/test_id_1000_v3.json'
    test_img_id_1000 = load_files(path_test_img_id_1000)

    while True:
        if args.interactive:
            vids = input("visual genome image id, comma separated : ")
            if int(vids) == -1:
                break
            elif int(vids) == 100:
                vids = test_img_id_1000[:100]
            elif int(vids) == 1000:
                vids = test_img_id_1000
            else:
                vids = vids.strip().split(',')
        else:
            vids = test_img_id_1000

        for vid in tqdm(vids):
            vg_id_infer = str(vid.strip())
            infer_sg = sg_test[vg_id_infer]
            word_vec_anchor = get_word_vec(infer_sg, infer_dset.vocab2idx, infer_dset.vocab_glove)
            HE_anchor, HEs_anchor = he_sampling(infer_sg['adj'], word_vec_anchor, infer_dset.sampling_steps, infer_dset.max_num_he, 'infer')
            resnet_dir = '/data/public/rw/datasets/visual_genome/Resnet_feature/wholeImg/'
            aux_anchor = np.load(os.path.join(resnet_dir, f'{vid}.npy'))
            HE_anchor = torch.tensor(HE_anchor).cuda()
            aux_anchor = torch.tensor(aux_anchor).cuda()

            # RESNET RERANKING
            l_target_id = get_resnet_rank(vid)
            target_ids = set(l_target_id)
            sg_ranked = {k: v for k, v in sg_test.items() if k in target_ids}

            rerank_infer_dset = Dset_VG_inference_AUX(model_cfg, sg_ranked, label_similarity, label_vg_ids,
                                                  infer_dset.vocab_glove, infer_dset.vocab2idx)
            infer_dloader = DataLoader(rerank_infer_dset, batch_size=10, num_workers=32, shuffle=False)


            infer_result = {}
            for b_idx, mini_batch in enumerate((infer_dloader)):
                # HE_compare, vg_id_compare = mini_batch
                score, att_map = prop_mini_batch_infer_aux(mini_batch, vid, HE_anchor, aux_anchor, infer_dset.label, infer_dset.label_id2idx, model)

                if args.visualize_att:
                    for idx_m in range(len(mini_batch[1])):
                        infer_result[mini_batch[1][idx_m]] = {"score": score[idx_m].item(), "att": att_map[idx_m].item()}
                else:
                    score_arr = [s.item() for s in score]
                    infer_result.update(list(zip(mini_batch[1], score_arr)))

            if args.visualize_att:
                save_json(infer_result, result_path + '/infer_result_w_att_epoch_{}_vid_{}.json'.format(num[-1], vid))
            else:
                data_pandas = pd.DataFrame.from_dict(infer_result, orient='index')
                data_pandas.to_csv(result_viewer_path+'/{}.tsv'.format(vid), sep='\t', header=False)#, index=False)
                save_json(infer_result, result_path + '/infer_result_w_att_epoch_{}_vid_{}.json'.format(num[-1], vid))

        if not args.interactive:
            break


def inference_AUX(model, sg_test, infer_dset, infer_dloader, args, model_cfg, label_similarity, label_vg_ids):
    ckpt_path = args.ckpt_path + args.exp_name
    ckpts = glob(os.path.join(ckpt_path, 'ckpt_*.pth.tar'))

    if len(ckpts) == 0:
        print("Error!, No saved models!")
        return -1

    num = []
    for ckpt in ckpts:
        tokens = ckpt.split('.')
        num.append(int(tokens[-3].split('_')[-1]))

    num.sort()
    last_ckpt = os.path.join(ckpt_path, 'ckpt_' + str(num[-1]) + '.pth.tar')
    print("Load the last model, epoch: %d" % (num[-1]))

    checkpoint = torch.load(last_ckpt)
    model.load_state_dict(checkpoint['state_dict'])
    print("loaded checkpoint '{}' (epoch {})".format(ckpts[-1], checkpoint['idx_epoch']))

    result_path = args.result_path + args.exp_name

    result_viewer_path = '/data/project/rw/viewer_CBIR/viewer/results/{}_epoch_{}'.format(args.exp_name, num[-1])
    if not os.path.exists(result_viewer_path):
        os.makedirs(result_viewer_path)

    path_test_img_id_1000 = '/data/project/rw/CBIR/test_id_1000_v3.json'
    test_img_id_1000 = load_files(path_test_img_id_1000)

    while True:
        if args.interactive:
            vids = input("visual genome image id, comma separated : ")
            if int(vids) == -1:
                break
            elif int(vids) == 100:
                vids = test_img_id_1000[:100]
            elif int(vids) == 1000:
                vids = test_img_id_1000
            else:
                vids = vids.strip().split(',')
        else:
            vids = test_img_id_1000

        for vid in tqdm(vids):
            vg_id_infer = str(vid.strip())
            infer_sg = sg_test[vg_id_infer]
            word_vec_anchor = get_word_vec(infer_sg, infer_dset.vocab2idx, infer_dset.vocab_glove)
            HE_anchor, HEs_anchor = he_sampling(infer_sg['adj'], word_vec_anchor, infer_dset.sampling_steps, infer_dset.max_num_he, 'infer')
            resnet_dir = '/data/public/rw/datasets/visual_genome/Resnet_feature/wholeImg/'
            aux_anchor = np.load(os.path.join(resnet_dir, f'{vid}.npy'))
            HE_anchor = torch.tensor(HE_anchor).cuda()
            aux_anchor = torch.tensor(aux_anchor).cuda()

            infer_result = {}
            for b_idx, mini_batch in enumerate((infer_dloader)):
                # HE_compare, vg_id_compare = mini_batch
                score, att_map = prop_mini_batch_infer_aux(mini_batch, vid, HE_anchor, aux_anchor, infer_dset.label, infer_dset.label_id2idx, model)

                if args.visualize_att:
                    for idx_m in range(len(mini_batch[1])):
                        infer_result[mini_batch[1][idx_m]] = {"score": score[idx_m].item(), "att": att_map[idx_m].item()}
                else:
                    score_arr = [s.item() for s in score]
                    infer_result.update(list(zip(mini_batch[1], score_arr)))

            if args.visualize_att:
                save_json(infer_result, result_path + '/infer_result_w_att_epoch_{}_vid_{}.json'.format(num[-1], vid))
            else:
                data_pandas = pd.DataFrame.from_dict(infer_result, orient='index')
                data_pandas.to_csv(result_viewer_path+'/{}.tsv'.format(vid), sep='\t', header=False)#, index=False)
                save_json(infer_result, result_path + '/infer_result_w_att_epoch_{}_vid_{}.json'.format(num[-1], vid))

        if not args.interactive:
            break



def get_resnet_rank(query_id, n_rank=100):
    resnet_result_path = '/data/project/rw/viewer_CBIR/viewer/results/test_resnet_cosine/'
    file_name = os.path.join(resnet_result_path, f'{query_id}.tsv')
    df = pd.read_csv(file_name, header=None, sep='\t')
    df = df.drop(index=df.index[df[0]==int(query_id)])  # remove self
    df = df.sort_values(1, ascending=False)[:n_rank]
    return [str(id_) for id_ in df[0]]


def main():
    ''' parse config file '''
    parser = argparse.ArgumentParser(description="Hypergraph Attention Networks for CBIR on VG dataset")

    parser.add_argument("--config_file", default="configs/han_sbert_tail_sampling_he_step_3.yaml")
    parser.add_argument("--exp_name", default="han_sbert_tail_6_in_100_HE_100_3")

    parser.add_argument("--trg_opt", type=str, default='SBERT')
    parser.add_argument("--resume", type=int, default=0)
    parser.add_argument("--inference", action='store_true')
    parser.add_argument("--instance", type=int, default=30)
    parser.add_argument("--visualize_att", action='store_true')
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--max_epoch", type=int, default=500)
    parser.add_argument("--tb_path", type=str, default='/data/project/rw/woong.ssang/CBIR/tb/')
    parser.add_argument("--result_path", type=str, default='/data/project/rw/woong.ssang/CBIR/results/')
    parser.add_argument("--ckpt_path", type=str, default='/data/project/rw/woong.ssang/CBIR/ckpt/')
    parser.add_argument("--rerank", action='store_true')
    parser.add_argument("--interactive", action='store_true')
    parser.add_argument("--epoch", type=int, default=None)
    args = parser.parse_args()

    model_cfg = load_files(args.config_file)
    print(model_cfg)

    if args.debug == False:
        summary_path = args.tb_path + args.exp_name
        summary = SummaryWriter(summary_path)

    result_path = args.result_path+args.exp_name
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    ckpt_path = args.ckpt_path + args.exp_name
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    # ----------- Load Dataset -----------------------------------------------
    tic = time.time()
    print("loading train data")
    sg_train = load_files(model_cfg['DATASET']['SG_TRAIN'])
    print("loaded train data {}s".format(time.time()-tic))

    tic = time.time()
    print("loading test data")
    sg_test = load_files(model_cfg['DATASET']['SG_TEST'])
    print("loaded test data {}s".format(time.time() - tic))
    vg_id_infer = list(sg_test.keys())[:args.instance]

    vocab_glove = load_files(model_cfg['DATASET']['VOCAB_GLOVE'])
    vocab2idx = load_files(model_cfg['DATASET']['VOCAB2IDX'])

    tic = time.time()
    print("loading label data")
    label_similarity_info = load_files(model_cfg['DATASET']['SIM_GT'])
    label_vg_ids = label_similarity_info['id']
    label_similarity = label_similarity_info['sims']
    print("loaded label data {}s".format(time.time()-tic))

    # ------------ Construct Dataset Class ------------------------------------
    train_dset = Dset_VG_Pairwise_AUX(model_cfg, sg_train, label_similarity, label_vg_ids, vocab_glove, vocab2idx, 'train')
    train_dloader = DataLoader(train_dset, batch_size=model_cfg['MODEL']['BATCH_SIZE'], num_workers=args.num_workers,
                           shuffle=True)
    test_dset = Dset_VG_Pairwise_AUX(model_cfg, sg_test, label_similarity, label_vg_ids, vocab_glove, vocab2idx, 'test')
    test_dloader = DataLoader(test_dset, batch_size=model_cfg['MODEL']['BATCH_SIZE'], num_workers=args.num_workers, shuffle=False)

    infer_dset = Dset_VG_inference_AUX(model_cfg, sg_test, label_similarity, label_vg_ids, vocab_glove, vocab2idx)
    infer_dloader = DataLoader(infer_dset, batch_size=model_cfg['MODEL']['BATCH_SIZE'], num_workers=args.num_workers, shuffle=False)

    # ------------ Model -----------------------
    model_from_yaml = model_cfg['MODEL'].get('MODEL', None)
    if model_from_yaml == 'HGAN_LATE_V1':
        model = HGAN_LATE_V1(model_cfg)
    elif model_from_yaml == 'HAN_AUX':
        model = HGAN_AUX(model_cfg)
    else:
        model = HGAN(model_cfg)

    optimizer = optim.Adam(model.parameters(), lr=model_cfg['MODEL']['LR'])
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[7, 15], gamma=0.5)

    model.cuda()
    resnet_weight = model_cfg['MODEL'].get('RESNET_WEIGHT', None)

    if args.inference == True:
        if args.rerank:
            if model_from_yaml is not None: 
                inference_rerank_AUX(model, sg_test, infer_dset, args, model_cfg, label_similarity, label_vg_ids)
            else:
                inference_rerank(model, sg_test, infer_dset, args, model_cfg, label_similarity, label_vg_ids)
        else:
            if model_from_yaml is not None:
                inference_AUX(model, sg_test, infer_dset, infer_dloader, args, model_cfg, label_similarity, label_vg_ids)
            else:
                inference(model, sg_test, infer_dset, infer_dloader, args, model_cfg, label_similarity, label_vg_ids)

        return 0
    # ------------ Iteration -----------------------
    train_loss = []
    num_iter = 0
    num_iter_test = 0
    test_loss = []
    for idx_epoch in range(args.max_epoch):
        # ------------ Training -----------------------
        torch.set_grad_enabled(True)
        model.train()

        for b_idx, mini_batch in enumerate(tqdm(train_dloader)):
            if model_cfg['MODEL']['TARGET'] == 'SBERT':
                score_p, loss = prop_mini_batch_sbert_aux(mini_batch, model, resnet_weight=resnet_weight)
            else:
                score_p, score_n, loss = prop_mini_batch(mini_batch, model)

            train_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.debug == False:
                summary.add_scalar('loss/train', loss.item(), num_iter)
            else:
                break
            num_iter += 1


        lr_scheduler.step()

        if args.debug == False:
            torch.save({'idx_epoch': idx_epoch,
                        'num_iter': num_iter,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                       os.path.join(ckpt_path, 'ckpt_%d.pth.tar' % (idx_epoch)))
            torch.save(model, os.path.join(ckpt_path, 'model.pth'))

        torch.set_grad_enabled(False)
        model.eval()

        for b_idx, mini_batch in enumerate(tqdm(test_dloader)):
            if model_cfg['MODEL']['TARGET'] == 'SBERT':
                score_p, loss = prop_mini_batch_sbert_aux(mini_batch, model, resnet_weight=resnet_weight)
            else:
                score_p, score_n, loss = prop_mini_batch(mini_batch, model)

            test_loss.append(loss.item())
            if args.debug == False:
                summary.add_scalar('loss/test', loss.item(), num_iter_test)
            else:
                break
            num_iter_test += 1

        """
        # ------------ Full Inference -----------------------
        #if idx_epoch > 0 and idx_epoch % 20 == 0:
        if idx_epoch % 20 == 0:
            infer_results = {}
            for idx_infer, vid in enumerate(vg_id_infer):
                print("INFERENCE, epoch: {}, num_infer: {}".format(idx_epoch, idx_infer))
                infer_sg = sg_test[vid]
                word_vec_anchor = get_word_vec(infer_sg, vocab2idx, vocab_glove)
                HE_anchor = he_sampling(infer_sg['adj'], word_vec_anchor, infer_dset.sampling_steps, infer_dset.max_num_he)
                infer_result = {}

                for b_idx, mini_batch in enumerate(tqdm(infer_dloader)):
                    # HE_compare, vg_id_compare = mini_batch
                    score, _ = prop_mini_batch_infer(mini_batch, vid, HE_anchor, infer_dset.label, infer_dset.label_id2idx, model)
                    score_arr = [s.item() for s in score]
                    infer_result.update( list(zip(mini_batch[1], score_arr)) )

                infer_results[vid] = infer_result

            if args.debug == False:
                save_json(infer_results, result_path+'/infer_result_epoch_{}.json'.format(idx_epoch))
        """

if __name__ == "__main__":
    main()
