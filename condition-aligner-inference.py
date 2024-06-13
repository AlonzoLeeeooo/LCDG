import torch
import json
import os
import random
import numpy as np
from omegaconf import OmegaConf

import torch.distributed as dist
import torch.multiprocessing as mp

import argparse
from condition_aligner_src import ConditionAdaptorRunner

def setup_seed(seed):
    print('Seed: ', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup(rank, world_size, master_port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = master_port

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print(f"Setting up the process on rank {rank}.")

def cleanup():
    dist.destroy_process_group()

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
    parser.add_argument("--mode", type=str, default='from_text', help='training mode: unconditional or from_text')
    parser.add_argument("-b", "--base", type=str, metavar="base_config.yaml", help="path of config file")
    parser.add_argument("-s", "--seed", type=int, default=23, help="setting seed")
    parser.add_argument("--get_files_from_path", action='store_true', help='get data from folder or flist')
    parser.add_argument("--indir", type=str, default='', help="input images dir")
    parser.add_argument("--text", type=str, default='', help="path prefix of text")
    parser.add_argument("--cond", type=str, default='', help='path prefix of condition')
    parser.add_argument("--outdir", type=str, default='', help="output dir")
    parser.add_argument("--inference_num", type=int, default=100, help='number of inference')
    parser.add_argument("--resume", type=str, help='resume from checkpoint')
    parser.add_argument("--master_port", type=str, default='12355', help='master port for setting up DDP process, default by 12355 as string type')
    parser.add_argument("--DDP", action='store_true', help='use ddp or not')

    return parser

# main function for DDP
def main_worker(rank, world_size, configs):
    setup(rank=rank, world_size=world_size, master_port=configs['args'].master_port)
    runner = ConditionAdaptorRunner(rank, configs=configs)
    with torch.no_grad():
        if configs['args'].mode == "from_text":
            runner.inference_from_text(configs)
        elif configs['args'].mode == "unconditional":
            runner.inference_from_unconditional(configs)

if __name__ == "__main__":
    
    print("\nHello World...\n")
    
    parser = get_parser()
    args = parser.parse_args()
    setup_seed(args.seed)
    
    args.inference = True
    
    # load config file
    model_configs = OmegaConf.load(args.base)
    configs = {'model_configs': model_configs, 'args': args}
    
    # prepare the logdir if not exist
    path = args.outdir
    os.makedirs(path, exist_ok=True)
    print('Output folder: %s' % (path))
        
    # multi GPU inference (no need)
    world_size = torch.cuda.device_count()
    
    if args.DDP:
        mp.spawn(main_worker, nprocs=world_size, args=(world_size, configs))
    else:
        runner = ConditionAdaptorRunner(rank=0, configs=configs)
        if configs['args'].mode == "from_text":
            runner.inference_from_text(configs)
        elif configs['args'].mode == "unconditional":
            runner.inference_from_unconditional(configs)