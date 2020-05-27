import argparse
import torch
import os
import sys
import json
import torch.optim as optim
from preprocess.utils import load_files, save_json
from dataloader import Dset_VG, Dset_VG_inference, Dset_VG_Pairwise, get_word_vec, get_sim, he_sampling, \
                       DsetImgPairwise, DsetSGPairwise, repeat_data, concat_data
from torch.utils.data import DataLoader
from torchvision import transforms
from model import HGAN, GraphEmbedding, HGAN_V0, HGAN_V2, HGAN_V3, HGAN_V4, GraphPool
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from glob import glob
import pandas as pd
from data import BERTSimilarity, get_reranked_ids, FlickrDataset, CocoDataset, VGDataset
from joblib import delayed, Parallel
from compute_ndcg import ndcg_batch


def split_id(l, split_idx, n_split):
    split_size = int(len(l) / n_split)
    if split_idx == n_split - 1:
        end = len(l)
    else:
        end = (split_idx + 1) * split_size
    start = split_idx * split_size
    print(start, end)
    return l[start:end]


def to_cuda(batch):
    if isinstance(batch, list):
        l = []
        for b in batch:
            if isinstance(b, dict):
                l.append({key: val.cuda() for key, val in b.items()})
            elif isinstance(b, torch.Tensor):
                l.append(b.cuda())
            else:
                raise ValueError
        return l
    elif isinstance(batch, dict):
        return {key: val.cuda() for key, val in batch.items()}
    elif isinstance(batch, torch.Tensor):
        return batch.cuda()
    else:
        raise ValueError


def inference(dataset_name, model, infer_dset, args):
    ckpt_path = args.ckpt_path + args.exp_name
    print(f'looking for checkpoints... {ckpt_path}')
    ckpts = glob(os.path.join(ckpt_path, 'ckpt_*.pth.tar'))

    if len(ckpts) == 0:
        print("Error!, No saved models!")
        return -1

    if args.epoch is None:
        num = []
        for ckpt in ckpts:
            tokens = ckpt.split('.')
            num.append(int(tokens[-3].split('_')[-1]))

        num.sort()
        loaded_epoch = num[-1]
    else:
        loaded_epoch = args.epoch
    last_ckpt = os.path.join(ckpt_path, 'ckpt_' + str(loaded_epoch) + '.pth.tar')
    print("Load the last model, epoch: %d" % (loaded_epoch))

    checkpoint = torch.load(last_ckpt)
    model.load_state_dict(checkpoint['state_dict'])
    print("loaded checkpoint '{}' (epoch {})".format(ckpts[-1], checkpoint['idx_epoch']))

    result_viewer_path = '/data/project/rw/viewer_CBIR/viewer/{}_results/{}_epoch_{}'.format(dataset_name, 
                                                                                             args.exp_name, loaded_epoch)
    if not os.path.exists(result_viewer_path):
        os.makedirs(result_viewer_path)

    if args.json and dataset_name == 'vg_coco':
        selected_id_file = 'test_id_1000_v3.json'
        with open(selected_id_file, 'r') as f:
            vids = json.load(f)
            vids = list(map(int, vids))
    else:
        vids = infer_dset.ds.d_split['test']  # test set image id
    infer_dset.sims = None

    if args.n_split is not None:
        vids = split_id(vids, args.split_idx, args.n_split)

    for vid in tqdm(vids):

        # get resnet rerank list
        l_reranked = get_reranked_ids(dataset_name, vid, n_rerank=100)

        # get image list
        time_s = time.time()
        l_imgs = Parallel(n_jobs=args.num_workers, prefer='threads')(delayed(infer_dset.get_by_id)(img_id) for img_id in l_reranked)
        target = concat_data(l_imgs)
        target = to_cuda(target)

        # get test image
        query = infer_dset.get_by_id(vid)
        query = repeat_data(query, len(l_imgs))
        query = to_cuda(query)

        # l_score = []
        # for batch in dl:
        #     batch = batch.cuda()
        #     with torch.no_grad():
        #         score = model(query, batch)
        #         l_score += score.flatten().detach().cpu().tolist()
        # score = l_score

        # run prediction
        # time_s = time.time()
        with torch.no_grad():
            score, _ = model.score(query, target)
            score = score.detach().cpu().flatten().tolist()
        infer_result = {'img_id': l_reranked, 'sim': score}
        # print(f'model: {time.time() - time_s} sec')

        # save result
        data_pandas = pd.DataFrame(infer_result)
        data_pandas[['img_id', 'sim']].to_csv(result_viewer_path+'/{}.tsv'.format(vid), sep='\t', header=False, index=False)
        # save_json(infer_result, result_path + '/infer_result_w_att_epoch_{}_vid_{}.json'.format(num[-1], vid))


def main():
    ''' parse config file '''
    parser = argparse.ArgumentParser(description="Hypergraph Attention Networks for CBIR on VG dataset")

    parser.add_argument("--config_file", default="configs/han_sbert_tail_sampling_he_step_3.yaml")
    parser.add_argument("--exp_name", default="han_sbert_tail_6_in_100_HE_100_3")
    parser.add_argument("--resume", type=int, default=0)
    parser.add_argument("--inference", action='store_true')
    parser.add_argument("--json", action='store_true', help='run inference on test_id_1000_v3.json file')
    parser.add_argument("--num_workers", type=int, default=28)
    parser.add_argument("--max_epoch", type=int, default=20)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--tb_path", type=str, default='/data/project/rw/woong.ssang/CBIR/tb/')
    parser.add_argument("--ckpt_path", type=str, default='/data/project/rw/woong.ssang/CBIR/ckpt/')
    parser.add_argument("--split_idx", type=int, default=None)
    parser.add_argument("--n_split", type=int, default=None)
    parser.add_argument("--test", action='store_true', help='for unitting the script')
    parser.add_argument("--infer_batch", action='store_true', help='use minibatch when computing NDCG')
    args = parser.parse_args()
    print(args)

    model_cfg = load_files(args.config_file)
    print(model_cfg)
    dataset_name = model_cfg['DATASET']['NAME']

    summary_path = args.tb_path + args.exp_name
    summary = SummaryWriter(summary_path)

    ckpt_path = args.ckpt_path + args.exp_name
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    # ----------- Load Dataset -----------------------------------------------
    tic = time.time()
    print("loading label data")
    if dataset_name == 'coco':
        sim_mat_file = '/data/project/rw/CBIR/data/coco/coco_sbert_mean.npy'
        sim_id_file = '/data/project/rw/CBIR/data/coco/coco_sbert_img_id.npy'
        sg_path = '/data/project/rw/CBIR/data/coco/coco_sgg_freq_prior_with_adj.pkl'
        vocab_emb_path = '/data/project/rw/CBIR/data/coco/glove_embs_coco_sgg_freq_prior.pkl' 
        vocab2idx_path = '/data/project/rw/CBIR/data/coco/vocab2idx_coco_sgg_freq_prior.pkl' 
        idx2vocab_path = '/data/project/rw/CBIR/data/coco/idx2vocab_coco_sgg_freq_prior.pkl'
        sims = BERTSimilarity(sim_mat_file, sim_id_file)
        ds = CocoDataset(vocab_emb=vocab_emb_path, vocab2idx=vocab2idx_path, idx2vocab=idx2vocab_path,
                         sg_path=sg_path)
    elif dataset_name == 'f30k':
        sim_mat_file = '/data/project/rw/CBIR/data/f30k/f30k_sbert_mean.npy'
        sim_id_file = '/data/project/rw/CBIR/data/f30k/f30k_sbert_img_id.npy'
        if model_cfg['DATASET']['TYPE'] == 'gen':
            sg_path = '/data/project/rw/CBIR/data/f30k/f30k_sgg_freq_prior_with_adj.pkl'
            vocab_emb_path = '/data/project/rw/CBIR/data/f30k/glove_embs_f30k_sgg_freq_prior.pkl' 
            vocab2idx_path = '/data/project/rw/CBIR/data/f30k/vocab2idx_f30k_sgg_freq_prior.pkl' 
            idx2vocab_path = '/data/project/rw/CBIR/data/f30k/idx2vocab_f30k_sgg_freq_prior.pkl'
        elif model_cfg['DATASET']['TYPE'] == 'grcnn':
            sg_path = '/data/project/rw/CBIR/data/f30k/f30k_sgg_grcnn_with_adj.pkl'
            vocab_emb_path = '/data/project/rw/CBIR/data/f30k/glove_embs_f30k_sgg_freq_prior.pkl' 
            vocab2idx_path = '/data/project/rw/CBIR/data/f30k/vocab2idx_f30k_sgg_freq_prior.pkl' 
            idx2vocab_path = '/data/project/rw/CBIR/data/f30k/idx2vocab_f30k_sgg_freq_prior.pkl'
        elif model_cfg['DATASET']['TYPE'] == 'butd':
            vocab_emb_path = '/data/project/rw/CBIR/data/f30k/butd_freq_prior_LR/glove_embs_f30k_sgg_butd_freq_prior_LR.pkl'
            vocab2idx_path = '/data/project/rw/CBIR/data/f30k/butd_freq_prior_LR/vocab2idx_f30k_sgg_butd_freq_prior_LR.pkl'
            idx2vocab_path = '/data/project/rw/CBIR/data/f30k/butd_freq_prior_LR/idx2vocab_f30k_sgg_butd_freq_prior_LR.pkl'
            sg_path = '/data/project/rw/CBIR/data/f30k/butd_freq_prior_LR/f30k_sgg_butd_freq_prior_LR_with_adj.pkl'
        sims = BERTSimilarity(sim_mat_file, sim_id_file)
        ds = FlickrDataset(vocab_emb=vocab_emb_path, vocab2idx=vocab2idx_path, idx2vocab=idx2vocab_path,
                           sg_path=sg_path)

    elif dataset_name == 'vg_coco':
        sim_mat_file = '/data/project/rw/CBIR/data/vg_coco/vg_coco_sbert_mean.npy'
        sim_id_file = '/data/project/rw/CBIR/data/vg_coco/vg_coco_sbert_img_id.npy'

        # default vocab files
        vocab_emb_path = '/data/project/rw/CBIR/data/vg_coco/glove_embs_vg_coco_sg_butd.pkl' 
        vocab2idx_path = '/data/project/rw/CBIR/data/vg_coco/vocab2idx_vg_coco_sg_butd.pkl' 
        idx2vocab_path = '/data/project/rw/CBIR/data/vg_coco/idx2vocab_vg_coco_sg_butd.pkl'
        if model_cfg['DATASET']['TYPE'] == 'GT':
            sg_path = '/data/project/rw/CBIR/data/vg_coco/vg_coco_gt_sg.pkl'
            vocab_emb_path = '/data/project/rw/CBIR/data/vg_coco/glove_embs_vg_coco_sg.pkl' 
            vocab2idx_path = '/data/project/rw/CBIR/data/vg_coco/vocab2idx_vg_coco_sg.pkl' 
            idx2vocab_path = '/data/project/rw/CBIR/data/vg_coco/idx2vocab_vg_coco_sg.pkl'
        elif model_cfg['DATASET']['TYPE'] == 'gen':
            sg_path = '/data/project/rw/CBIR/data/vg_coco/vg_coco_sgg.pkl'
        elif model_cfg['DATASET']['TYPE'] == 'noatt':
            sg_path = '/data/project/rw/CBIR/data/vg_coco/vg_coco_sgg_noatt.pkl'
        elif model_cfg['DATASET']['TYPE'] == 'norel':
            sg_path = '/data/project/rw/CBIR/data/vg_coco/vg_coco_sgg_norel.pkl'
        elif model_cfg['DATASET']['TYPE'] == 'norelsym':
            sg_path = '/data/project/rw/CBIR/data/vg_coco/vg_coco_sgg_norelsym.pkl'
        elif model_cfg['DATASET']['TYPE'] == 'randrel':
            sg_path = '/data/project/rw/CBIR/data/vg_coco/vg_coco_sgg_randrel.pkl'
        elif model_cfg['DATASET']['TYPE'] == 'obj':
            sg_path = '/data/project/rw/CBIR/data/vg_coco/vg_coco_sgg_obj.pkl'
        elif model_cfg['DATASET']['TYPE'] == 'grcnn':
            sg_path = '/data/project/rw/CBIR/data/vg_coco/sgg_grcnn/vgcoco_sgg_grcnn_with_adj.pkl'
            vocab_emb_path = '/data/project/rw/CBIR/data/vg_coco/sgg_grcnn/glove_embs_vgcoco_sgg_grcnn.pkl' 
            vocab2idx_path = '/data/project/rw/CBIR/data/vg_coco/sgg_grcnn/vocab2idx_vgcoco_sgg_grcnn.pkl' 
            idx2vocab_path = '/data/project/rw/CBIR/data/vg_coco/sgg_grcnn/idx2vocab_vgcoco_sgg_grcnn.pkl'
        elif model_cfg['DATASET']['TYPE'] == 'GTconn':  # connected GT graph
            sg_path = '/data/project/rw/txt-img-retrieval/data/coco_sgalign/sg_align/vgcoco_sgg_newgt_connected_all_revised_with_adj.pkl'
            vocab_emb_path = '/data/project/rw/txt-img-retrieval/data/coco_sgalign/glove/glove_embs_vgcoco_sgg_newgt_connected_all_revised.pkl'
            vocab2idx_path = '/data/project/rw/txt-img-retrieval/data/coco_sgalign/glove/vocab2idx_vgcoco_sgg_newgt_connected_all_revised.pkl'
            idx2vocab_path = '/data/project/rw/txt-img-retrieval/data/coco_sgalign/glove/idx2vocab_vgcoco_sgg_newgt_connected_all_revised.pkl'

        print(f'scene graph file: {sg_path}')
        print(f'vocab embedding: {vocab_emb_path}')
        print(f'vocab2idx : {vocab2idx_path}')
        print(f'idx2vocab : {idx2vocab_path}')

        sims = BERTSimilarity(sim_mat_file, sim_id_file)
        if model_cfg['DATASET']['TYPE'] == 'GTconn':  # connected GT graph
            ds = VGDataset(vocab_emb=vocab_emb_path, vocab2idx=vocab2idx_path, idx2vocab=idx2vocab_path,
                           sg_path=sg_path, new_split=True)
        else:
            ds = VGDataset(vocab_emb=vocab_emb_path, vocab2idx=vocab2idx_path, idx2vocab=idx2vocab_path,
                           sg_path=sg_path, new_split=False)
    print("loaded label data {}s".format(time.time()-tic))

    # ------------ Construct Dataset Class ------------------------------------
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224),
                                    transforms.ToTensor(), normalize])

    mcfg = model_cfg['MODEL']
    if args.inference:
        infer_dset = DsetSGPairwise(ds, sims, tail_range=mcfg['TAIL_RANGE'], split='test',
                                    mode=mcfg['MODE'], sample_mode=mcfg['SAMPLE_MODE'],
                                    num_steps=mcfg['STEP'], max_num_he=mcfg['NUM_MAX_HE'],
                                    bbox_size=mcfg.get('bbox_size', 64),
                                    pos_k=mcfg.get('POS_K', None))
    else:
        if dataset_name == 'vg_coco':
            val_split = 'test'
        else:
            val_split = 'val'

        train_dset = DsetSGPairwise(ds, sims, tail_range=mcfg['TAIL_RANGE'], split='train',
                                    mode=mcfg['MODE'], sample_mode=mcfg['SAMPLE_MODE'],
                                    num_steps=mcfg['STEP'], max_num_he=mcfg['NUM_MAX_HE'],
                                    bbox_size=mcfg.get('bbox_size', 64),
                                    pos_k=mcfg.get('POS_K', None))
        train_dloader = DataLoader(train_dset, batch_size=mcfg['BATCH_SIZE'], num_workers=args.num_workers,
                                   shuffle=True, collate_fn=concat_data)
        test_dset = DsetSGPairwise(ds, sims, tail_range=mcfg['TAIL_RANGE'], split=val_split,
                                   mode=mcfg['MODE'], sample_mode=mcfg['SAMPLE_MODE'],
                                   num_steps=mcfg['STEP'], max_num_he=mcfg['NUM_MAX_HE'],
                                   bbox_size=mcfg.get('bbox_size', 64),
                                   pos_k=mcfg.get('POS_K', None))
        test_dloader = DataLoader(test_dset, batch_size=mcfg['BATCH_SIZE'], num_workers=args.num_workers, shuffle=False,
                                  collate_fn=concat_data)

    # ------------ Model -----------------------
    model_name = mcfg.get('NAME', 'HAN')
    if model_name == 'GraphEmbedding':
        model = GraphEmbedding(model_cfg)
    elif model_name == 'GraphPool':
        model = GraphPool(model_cfg)
    elif model_name == 'HGAN_V0':
        model = HGAN_V0(model_cfg)
    elif model_name == 'HGAN_V2':
        model = HGAN_V2(model_cfg)
    elif model_name == 'HGAN_V3':
        model = HGAN_V3(model_cfg)
    elif model_name == 'HGAN_V4':
        model = HGAN_V4(model_cfg)
    else:
        print('model: HAN')
        model = HGAN(model_cfg)
    model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=model_cfg['MODEL']['LR'])
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[7, 15], gamma=0.5)


    if args.inference == True:
        inference(dataset_name, model, infer_dset, args)
        return 0
    # ------------ Iteration -----------------------
    l_val_ids = sorted(test_dset.l_id)[:10]
    ks = (5, 10, 20)
    train_loss = []
    num_iter = 0
    num_iter_test = 0
    test_loss = []
    for idx_epoch in range(args.max_epoch):
        # ------------ Training -----------------------
        torch.set_grad_enabled(True)
        model.train()

        for b_idx, mini_batch in enumerate(tqdm(train_dloader)):
            mini_batch = to_cuda(mini_batch)
            optimizer.zero_grad()
            pred, loss = model(*mini_batch)
            if isinstance(loss, dict):
                d_loss = loss
                loss = loss['loss']

            train_loss.append(loss.item())

            loss.backward()
            optimizer.step()

            summary.add_scalar('loss/train', loss.item(), num_iter)
            if model_name == 'GraphPool':
                summary.add_scalar('loss/entrloss', d_loss['entrloss'].item(), num_iter)
                summary.add_scalar('loss/linkloss', d_loss['linkloss'].item(), num_iter)
                summary.add_scalar('loss/specloss', d_loss['specloss'].item(), num_iter)
            num_iter += 1

            if args.test:
                break

        lr_scheduler.step()

        torch.save({'idx_epoch': idx_epoch,
                    'num_iter': num_iter,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()},
                   os.path.join(ckpt_path, 'ckpt_%d.pth.tar' % (idx_epoch)))
        torch.save(model, os.path.join(ckpt_path, 'model.pth'))

        torch.set_grad_enabled(False)
        model.eval()

        # --------- validation ---------
        # # validation loss
        # for b_idx, mini_batch in enumerate(tqdm(test_dloader)):
        #     mini_batch = to_cuda(mini_batch)
        #     pred, loss = model(*mini_batch)

        #     test_loss.append(loss.item())
        #     summary.add_scalar('loss/test', loss.item(), num_iter_test)
        #     num_iter_test += 1
        #     if args.test:
        #         break

        # validation ndcg
        l_pred = []
        l_true_rel = []
        for val_id in tqdm(l_val_ids):
            l_reranked = get_reranked_ids(dataset_name, val_id, n_rerank=100, split=val_split)
            l_imgs = Parallel(n_jobs=5, prefer='threads')(delayed(test_dset.get_by_id)(img_id) for img_id in l_reranked)

            if args.infer_batch:
                query = test_dset.get_by_id(val_id)
                query = repeat_data(query, 10)
                query = to_cuda(query)

                l_score = []
                for i_batch in range(10):
                    target = concat_data(l_imgs[i_batch * 10:(i_batch+1) * 10])
                    target = to_cuda(target)

                    with torch.no_grad():
                        score, _ = model.score(query, target)
                        score = score.detach().cpu().flatten().tolist()
                        l_score += score
                score = l_score
            else:
                query = test_dset.get_by_id(val_id)
                query = repeat_data(query, len(l_imgs))
                query = to_cuda(query)
                target = concat_data(l_imgs)
                target = to_cuda(target)

                query = test_dset.get_by_id(val_id)
                query = repeat_data(query, len(l_imgs))
                query = to_cuda(query)
                with torch.no_grad():
                    score, _ = model.score(query, target)
                    score = score.detach().cpu().flatten().tolist()

            l_rel = [sims.get_similarity(val_id, id_) for id_ in l_reranked]
            l_pred.append(score)
            l_true_rel.append(l_rel)
            if args.test:
                break
        l_true_rel = torch.tensor(l_true_rel, dtype=torch.float)
        l_pred = torch.tensor(l_pred, dtype=torch.float)
        val_ndcg = ndcg_batch(l_pred, l_true_rel, ks=ks)
        for k in ks:
            val_ndcg[k] = val_ndcg[k].mean().item()
        print(f'epoch: {idx_epoch}, {val_ndcg}')
        for k in ks:
            summary.add_scalar(f'ndcg/{k}', val_ndcg[k], idx_epoch+1)

if __name__ == "__main__":
    main()
