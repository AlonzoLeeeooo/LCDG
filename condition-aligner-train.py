import torch
import json
import os
import random
import numpy as np
from omegaconf import OmegaConf

import torch.distributed as dist
import torch.multiprocessing as mp

import argparse
from condition_aligner_src import ConditionAlignerRunner

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
if device == "cpu":
    print("Warning: you are using cpu to train, fool!")

# initialize DDP parameters
def setup(rank, world_size, master_port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = master_port

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print(f"Setting up the process on rank {rank}.")

def cleanup():
    dist.destroy_process_group()

def setup_seed(seed):
    print('Seed: ', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False 
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-r", "--resume", type=str, const=True, default="", nargs="?", help="resume from previous checkpoint")
    parser.add_argument("-b", "--base", type=str, metavar="base_config.yaml", help="file folder of config file")
    parser.add_argument("-s", "--seed", type=int, default=23, help="setting seed")
    parser.add_argument("-l", "--logdir", type=str, default="outputs/verbose/training", help="directory for logging, saving checkpoints and tensorboard logs")
    parser.add_argument("--verbose", action='store_true', help='turn on verbose mode')
    parser.add_argument("--DDP", action='store_true', help='use ddp or not')
    parser.add_argument("--master_port", type=str, default='12355', help='master port for setting up DDP process, default by 12355 as string type')
    
    # training paramters
    parser.add_argument("--batch_size", type=int, default=4, help='batch size')
    parser.add_argument("--num_workers", type=int, default=8, help='number of threads loading data')
    parser.add_argument("--epochs", type=int, default=100, help='maximum training epochs, UNUSED')
    parser.add_argument("--max_steps", type=int, default=30001, help='maximum training steps')
    parser.add_argument("--print_freq", type=int, default=1, help='the frequency of printing logs info')
    parser.add_argument("--sample_freq", type=int, default=1000, help='the frequency of saving samples')
    parser.add_argument("--checkpoint_freq", type=int, default=10000, help='the frequency of saving checkpoints')
    parser.add_argument("--eval_freq", type=int, default=0, help='the frequency of evalution, 0 represents not using evalution')
 
    return parser


# main function for DDP
def main_worker(rank, world_size, configs):
    setup(rank=rank, world_size=world_size, master_port=configs['args'].master_port)
    runner = ConditionAlignerRunner(rank, configs=configs)
    
    if configs['args'].mode == "from_text":
        runner.train_from_text_condition()
    elif configs['args'].mode == "unconditional":
        runner.train_from_unconditional()
    

if __name__ == '__main__':
    print("\nHello World...\n")
    
    parser = get_parser()

    args = parser.parse_args()
    setup_seed(args.seed)

    args.evaluate = False
    args.inference = False
    
    # load config file
    model_configs = OmegaConf.load(args.base)
    
    mode = model_configs['condition_aligner_config']['mode']
    args.use_style_loss = True if model_configs['condition_aligner_config']['cond_type'] == "style" else False
    
    # prepare the logdir if not exist
    path = args.logdir
    os.makedirs(path, exist_ok=True)
    print('\nExperiment folder created at: %s.\n' % (path))

    configs = {'args': args, 'model_configs': model_configs}
    
    # setup basic DDP parameters
    args.world_size = torch.cuda.device_count()
    
    # using DDP to train
    if args.DDP:
        mp.spawn(main_worker, nprocs=args.world_size, args=(args.world_size, configs))
    # using Mac MPS to train 
    elif device == "mps":
        runner = ConditionAlignerRunner(rank=0, configs=configs)
        runner.T2I_training()
    else:
        # define runner
        runner = ConditionAlignerRunner(rank=0, configs=configs)
        if mode == "from_text":
            runner.T2I_model_training()
        elif mode == "unconditional":
            runner.unconditional_model_training()
        else:
            raise NotImplementedError ("You need to set ``mode'' in the configuration file!")