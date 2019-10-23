import argparse
import torch
import os
import torch.optim as optim
from preprocess.utils import load_files
from dataloader import Dset_VG
from torch.utils.data import DataLoader
from model import HGAN
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

"""
def train(cfg, args):

def test(cfg, args):
"""

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
    parser.add_argument("--exp_name", default="han_baseline")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--resume", type=int, default=0)
    parser.add_argument("--inference", action='store_true')
    parser.add_argument("--instance", type=int, default=-1)
    parser.add_argument("--visualize", action='store_true')
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--tb_path", type=str, default='/data/project/rw/CBIR/tb/')
    parser.add_argument("--result_path", type=str, default='/data/project/rw/Charades_ori/Charades_SGG')
    parser.add_argument("--ckpt_path", type=str, default='/data/project/rw/VG_eskim/checkpoints/baseline_freq')
    args = parser.parse_args()

    model_cfg = load_files(args.config_file)
    print(model_cfg)

    if args.debug == False:
        summary_path = args.tb_path + args.exp_name
        summary = SummaryWriter(summary_path)

    # ----------- Load Dataset -----------------------------------------------
    tic = time.time()
    print("loading train data")
    sg_train = load_files(model_cfg['DATASET']['SG_TRAIN'])
    print("loaded train data {}s".format(time.time()-tic))

    tic = time.time()
    print("loading test data")
    sg_test = load_files(model_cfg['DATASET']['SG_TEST'])
    print("loaded test data {}s".format(time.time() - tic))

    vocab_glove = load_files(model_cfg['DATASET']['VOCAB_GLOVE'])
    vocab2idx = load_files(model_cfg['DATASET']['VOCAB2IDX'])

    tic = time.time()
    print("loading label data")
    label_similarity_info = load_files(model_cfg['DATASET']['SIM_GT'])
    label_vg_ids = label_similarity_info['id']
    label_similarity = label_similarity_info['sims']
    print("loaded label data {}s".format(time.time()-tic))

    # ------------ Construct Dataset Class ------------------------------------
    train_dset = Dset_VG(model_cfg, sg_train, label_similarity, label_vg_ids, vocab_glove, vocab2idx, 'train')
    train_dloader = DataLoader(train_dset, batch_size=model_cfg['MODEL']['BATCH_SIZE'], num_workers=args.num_workers, shuffle=True)

    test_dset = Dset_VG(model_cfg, sg_test, label_similarity, label_vg_ids, vocab_glove, vocab2idx, 'test')
    test_dloader = DataLoader(test_dset, batch_size=model_cfg['MODEL']['BATCH_SIZE'], num_workers=1, shuffle=False)

    # ------------ Model -----------------------
    model = HGAN(model_cfg)

    optimizer = optim.Adam(model.parameters(), lr=model_cfg['MODEL']['LR'])
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[7, 15], gamma=0.5)

    model.cuda()

    """
    # ------------ misc. -----------------------
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        #synchronize()
    """
    # ------------ Training -----------------------

    train_loss = []
    num_iter = 0

    for idx_epoch in range(args.max_epoch):
        torch.set_grad_enabled(True)
        model.train()

        for b_idx, mini_batch in enumerate(tqdm(train_dloader)):
            score_p, score_n, loss = prop_mini_batch(mini_batch, model)
            train_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.debug == False:
                summary.add_scalar('loss/train', loss.item(), num_iter)

            num_iter += 1
        lr_scheduler.step()

    """
    if not args.inference:
        model = train(cfg, args)
    else:
        test(cfg, args)
    """

if __name__ == "__main__":
    main()