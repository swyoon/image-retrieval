import subprocess

def run_train():
    #subprocess.call(['python', '-m', 'torch.distributed.launch', '--nproc_per_node','2',
    #'main.py', '--config-file', 'configs/sgg_res101_joint.yaml', '--algorithm', 'sg_baseline'])
    subprocess.call([
        'python', 'main.py', '--config_file', 'configs/han_baseline.yaml',
        '--inference',
        '--debug'
    ])

if __name__ == "__main__":
    run_train()