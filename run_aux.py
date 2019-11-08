import subprocess
import os

def run_train():
    #subprocess.call(['python', '-m', 'torch.distributed.launch', '--nproc_per_node','2',
    #'main.py', '--config-file', 'configs/sgg_res101_joint.yaml', '--algorithm', 'sg_baseline'])
    subprocess.call([
        'python', 'aux_main.py',
        '--config_file', 'configs/han_sbert_tail_200_3_he_200_5_aux.yaml',
        '--exp_name', 'aux2',
        '--inference',
        '--rerank',
        #'--debug'
    ])

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    run_train()
