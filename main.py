import argparse
import torch
import os
import torch.optim as optim
from preprocess.utils import load_files, save_json
from dataloader import Dset_VG, Dset_VG_inference, Dset_VG_Pairwise, get_word_vec, get_sim, he_sampling
from torch.utils.data import DataLoader
from model import HGAN
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


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

        score, loss = model(batch_anchor, HE_compare, sim_score)
    else:
        score, _, loss = model(batch_anchor, HE_compare, HE_compare)
    return score


def prop_mini_batch_sbert(mini_batch, model):
    HE_a, HE_p, sbert_score = mini_batch
    HE_a = HE_a.cuda()
    HE_p = HE_p.cuda()
    sbert_score = sbert_score.cuda()

    score_p, loss = model(HE_a, HE_p, sbert_score)

    return score_p, loss

def prop_mini_batch(mini_batch, model):
    HE_a, HE_p, HE_n = mini_batch
    HE_a = HE_a.cuda()
    HE_p = HE_p.cuda()
    HE_n = HE_n.cuda()

    score_p, score_n, loss = model(HE_a, HE_p, HE_n)

    return score_p, score_n, loss


def main():
    ''' parse config file '''
    parser = argparse.ArgumentParser(description="Hypergraph Attention Networks for CBIR on VG dataset")
    parser.add_argument("--config_file", default="configs/han_baseline.yaml")
    parser.add_argument("--exp_name", default="han_sbert_regression")
    parser.add_argument("--trg_opt", type=str, default='SBERT')
    parser.add_argument("--resume", type=int, default=0)
    parser.add_argument("--inference", action='store_true')
    parser.add_argument("--instance", type=int, default=30)
    parser.add_argument("--visualize", action='store_true')
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--tb_path", type=str, default='/data/project/rw/CBIR/tb/')
    parser.add_argument("--result_path", type=str, default='/data/project/rw/CBIR/results/')
    parser.add_argument("--ckpt_path", type=str, default='/data/project/rw/CBIR/ckpt/')
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
    if model_cfg['MODEL']['TARGET'] == 'SBERT':
        train_dset = Dset_VG_Pairwise(model_cfg, sg_train, sg_test, label_similarity, label_vg_ids, vocab_glove, vocab2idx, 'train')
        test_dset = Dset_VG_Pairwise(model_cfg, sg_train, sg_test, label_similarity, label_vg_ids, vocab_glove, vocab2idx, 'test')
        test_dloader = DataLoader(test_dset, batch_size=model_cfg['MODEL']['BATCH_SIZE'], num_workers=args.num_workers, shuffle=False)
    else:
        train_dset = Dset_VG(model_cfg, sg_train, label_similarity, label_vg_ids, vocab_glove, vocab2idx)

    train_dloader = DataLoader(train_dset, batch_size=model_cfg['MODEL']['BATCH_SIZE'], num_workers=args.num_workers, shuffle=True)

    infer_dset = Dset_VG_inference(model_cfg, sg_train, label_similarity, label_vg_ids, vocab_glove, vocab2idx)
    infer_dloader = DataLoader(infer_dset, batch_size=model_cfg['MODEL']['BATCH_SIZE'], num_workers=args.num_workers, shuffle=False)

    # ------------ Model -----------------------
    model = HGAN(model_cfg)

    optimizer = optim.Adam(model.parameters(), lr=model_cfg['MODEL']['LR'])
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[7, 15], gamma=0.5)

    model.cuda()

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
                score_p, loss = prop_mini_batch_sbert(mini_batch, model)
            else:
                score_p, score_n, loss = prop_mini_batch(mini_batch, model)

            train_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.debug == False:
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

        torch.set_grad_enabled(False)
        model.eval()

        for b_idx, mini_batch in enumerate(tqdm(test_dloader)):
            if model_cfg['MODEL']['TARGET'] == 'SBERT':
                score_p, loss = prop_mini_batch_sbert(mini_batch, model)
            else:
                score_p, score_n, loss = prop_mini_batch(mini_batch, model)

            test_loss.append(loss.item())
            if args.debug == False:
                summary.add_scalar('loss/test', loss.item(), num_iter_test)
            num_iter_test += 1
            break

        # ------------ Full Inference -----------------------
        if idx_epoch % 10 == 0:
            infer_results = {}
            for idx_infer, vid in enumerate(vg_id_infer):
                print("INFERENCE, epoch: {}, num_infer: {}".format(idx_epoch, idx_infer))
                infer_sg = sg_test[vid]
                word_vec_anchor = get_word_vec(infer_sg, vocab2idx, vocab_glove)
                HE_anchor = he_sampling(infer_sg['adj'], word_vec_anchor, infer_dset.sampling_steps, infer_dset.max_num_he)
                infer_result = {}

                for b_idx, mini_batch in enumerate(tqdm(infer_dloader)):
                    # HE_compare, vg_id_compare = mini_batch
                    score = prop_mini_batch_infer(mini_batch, vid, HE_anchor, infer_dset.label, infer_dset.label_id2idx, model)
                    score_arr = [s.item() for s in score]
                    infer_result.update( list(zip(vg_id_compare, score_arr)) )

                infer_results[vid] = infer_result

            save_json(infer_results, result_path+'/infer_result_epoch_{}.json'.format(idx_epoch))


if __name__ == "__main__":
    main()