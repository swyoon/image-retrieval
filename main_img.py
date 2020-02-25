import argparse
import torch
import os
import sys
import torch.optim as optim
from preprocess.utils import load_files, save_json
from dataloader import Dset_VG, Dset_VG_inference, Dset_VG_Pairwise, get_word_vec, get_sim, he_sampling, \
                       DsetImgPairwise
from torch.utils.data import DataLoader
from torchvision import transforms
from model import DeepMetric
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from glob import glob
import pandas as pd
from data import BERTSimilarity, get_reranked_ids
from joblib import delayed, Parallel
from compute_ndcg import ndcg_batch


def inference(dataset_name, model, infer_dset, args):
    ckpt_path = args.ckpt_path + args.exp_name
    print(f'looking for checkpoints... {ckpt_path}')
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

    result_viewer_path = '/data/project/rw/viewer_CBIR/viewer/{}_results/{}_epoch_{}'.format(dataset_name, 
                                                                                             args.exp_name, num[-1])
    if not os.path.exists(result_viewer_path):
        os.makedirs(result_viewer_path)

    # path_test_img_id_1000 = '/data/project/rw/CBIR/test_id_1000_v2.json'
    # test_img_id_1000 = load_files(path_test_img_id_1000)
    vids = infer_dset.ds.d_split['test']  # test set image id
    infer_dset.sims = None

    for vid in tqdm(vids):

        # get resnet rerank list
        l_reranked = get_reranked_ids(dataset_name, vid, n_rerank=100)
        # infer_dset.set_iter_ids(l_reranked)
        # dl = DataLoader(infer_dset, shuffle=False, num_workers=5, batch_size=20)


        # get image list
        # time_s = time.time()
        # target = torch.stack([infer_dset.get_by_id(img_id) for img_id in l_reranked])
        l_imgs = Parallel(n_jobs=10, prefer='threads')(delayed(infer_dset.get_by_id)(img_id) for img_id in l_reranked)
        # target = torch.stack([infer_dset.get_by_id(img_id) for img_id in l_reranked])
        target = torch.stack(l_imgs)
        target = target.cuda()
        # print(f'image loading {time.time() - time_s} sec')

        # get test image
        query = infer_dset.get_by_id(vid)
        # query = query.unsqueeze(0).repeat(len(target), 1, 1, 1)
        query = query.unsqueeze(0)
        query = query.cuda()
        # from pudb import set_trace; set_trace()

        # l_score = []
        # for batch in dl:
        #     batch = batch.cuda()
        #     with torch.no_grad():
        #         score = model(query, batch)
        #         l_score += score.flatten().detach().cpu().tolist()
        # score = l_score

        # run prediction
        with torch.no_grad():
            score = model(query, target)
            score = score.detach().cpu().flatten().tolist()
        infer_result = {'img_id': l_reranked, 'sim': score}

        # save result
        data_pandas = pd.DataFrame(infer_result)
        data_pandas[['img_id', 'sim']].to_csv(result_viewer_path+'/{}.tsv'.format(vid), sep='\t', header=False, index=False)
        # save_json(infer_result, result_path + '/infer_result_w_att_epoch_{}_vid_{}.json'.format(num[-1], vid))



def main():
    ''' parse config file '''
    parser = argparse.ArgumentParser(description="Hypergraph Attention Networks for CBIR on VG dataset")

    parser.add_argument("--config_file", default="configs/han_sbert_tail_sampling_he_step_3.yaml")
    parser.add_argument("--exp_name", default="han_sbert_tail_6_in_100_HE_100_3")
    parser.add_argument("--trg_opt", type=str, default='SBERT')
    parser.add_argument("--resume", type=int, default=0)
    parser.add_argument("--inference", action='store_true')
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--max_epoch", type=int, default=500)
    parser.add_argument("--tb_path", type=str, default='/data/project/rw/CBIR/tb/')
    parser.add_argument("--result_path", type=str, default='/data/project/rw/CBIR/results/')
    parser.add_argument("--ckpt_path", type=str, default='/data/project/rw/CBIR/ckpt/')
    args = parser.parse_args()

    model_cfg = load_files(args.config_file)
    print(model_cfg)
    dataset_name = model_cfg['DATASET']['NAME']

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
    print("loading label data")
    if dataset_name == 'coco':
        sim_mat_file = '/data/project/rw/CBIR/data/coco/coco_sbert_mean.npy'
        sim_id_file = '/data/project/rw/CBIR/data/coco/coco_sbert_img_id.npy'
    elif dataset_name == 'f30k':
        sim_mat_file = '/data/project/rw/CBIR/data/f30k/f30k_sbert_mean.npy'
        sim_id_file = '/data/project/rw/CBIR/data/f30k/f30k_sbert_img_id.npy'
    elif dataset_name == 'vg_coco':
        sim_mat_file = '/data/project/rw/CBIR/data/vg_coco/vg_coco_sbert_mean.npy'
        sim_id_file = '/data/project/rw/CBIR/data/vg_coco/vg_coco_sbert_img_id.npy'
    sims = BERTSimilarity(sim_mat_file, sim_id_file)
    print("loaded label data {}s".format(time.time()-tic))

    # ------------ Construct Dataset Class ------------------------------------
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224),
                                    transforms.ToTensor(), normalize])
    if args.inference:
        infer_dset = DsetImgPairwise(dataset_name, sims, tail_range=model_cfg['MODEL']['TAIL_RANGE'], split='test',
                                     transforms=transform)
    else:
        train_dset = DsetImgPairwise(dataset_name, sims, tail_range=model_cfg['MODEL']['TAIL_RANGE'], split='train',
                                     transforms=transform, sample_mode=model_cfg['MODEL']['SAMPLE_MODE'])
        train_dloader = DataLoader(train_dset, batch_size=model_cfg['MODEL']['BATCH_SIZE'], num_workers=args.num_workers,
                                   shuffle=True)
        if dataset_name == 'vg_coco':
            val_split = 'test'
        else:
            val_split = 'val'
        test_dset = DsetImgPairwise(dataset_name, sims, tail_range=model_cfg['MODEL']['TAIL_RANGE'], split=val_split,
                                    transforms=transform)
        test_dloader = DataLoader(test_dset, batch_size=model_cfg['MODEL']['BATCH_SIZE'], num_workers=args.num_workers, shuffle=False)

    # ------------ Model -----------------------
    d_model = model_cfg['MODEL']
    model = DeepMetric(backbone='resnet152', finetune=d_model['FINETUNE'],
                       normalize=d_model['NORMALIZE'], mode=d_model['MODE'], margin=d_model['MARGIN'],
                       embedding_dim=d_model['EMBEDDING_DIM'])
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
            mini_batch = [m.cuda() for m in mini_batch]
            optimizer.zero_grad()
            loss = model.loss(*mini_batch)

            train_loss.append(loss.item())

            loss.backward()
            optimizer.step()

            summary.add_scalar('loss/train', loss.item(), num_iter)
            num_iter += 1
            break

        lr_scheduler.step()

        torch.save({'idx_epoch': idx_epoch,
                    'num_iter': num_iter,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()},
                   os.path.join(ckpt_path, 'ckpt_%d.pth.tar' % (idx_epoch)))
        torch.save(model, os.path.join(ckpt_path, 'model.pth'))

        # ---------- validation -------------
        torch.set_grad_enabled(False)
        model.eval()

        # validation loss
        for b_idx, mini_batch in enumerate(tqdm(test_dloader)):
            mini_batch = [m.cuda() for m in mini_batch]
            loss = model.loss(*mini_batch)

            test_loss.append(loss.item())
            summary.add_scalar('loss/test', loss.item(), num_iter_test)
            num_iter_test += 1

        # validation ndcg
        l_pred = []
        l_true_rel = []
        for val_id in tqdm(l_val_ids):
            l_reranked = get_reranked_ids(dataset_name, val_id, n_rerank=100, split=val_split)

            l_imgs = Parallel(n_jobs=10, prefer='threads')(delayed(test_dset.get_by_id)(img_id) for img_id in l_reranked)
            target = torch.stack(l_imgs)
            target = target.cuda()
            query = test_dset.get_by_id(val_id)
            query = query.unsqueeze(0)
            query = query.cuda()
            with torch.no_grad():
                score = model(query, target)
                score = score.detach().cpu().flatten().tolist()
            l_rel = [sims.get_similarity(val_id, id_) for id_ in l_reranked]
            l_pred.append(score)
            l_true_rel.append(l_rel)
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
