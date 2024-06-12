import argparse, os, sys
from omegaconf import OmegaConf
import numpy as np
import cv2
import json
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist

from condition_aligner_src import ConditionAlignerRunner

DEFAULT_NEGATIVE_PROMPT = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, ' \
                          'fewer digits, cropped, worst quality, low quality'

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
def setup_seed(seed):
    print('Seed: ', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

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
    parser.add_argument("--indir", type=str, default='DATA_FILELIST_PATH', help="input images dir")
    parser.add_argument("--text", type=str, default='CAPTION_PATH', help="text prefix")
    parser.add_argument("--target_cond", type=str, default='VALIDATION_DATA_PATH', help='edge prefix')
    parser.add_argument("--outdir", type=str, default='outputs/verbose/sampling', help="output dir")
    parser.add_argument("--resume", type=str, help='checkpoint of edge aligner')
    parser.add_argument("--steps", type=int, default=50, help='number of ddim sampling steps')
    parser.add_argument("--cond_scale", type=float, default=2.0, help='scale of condition score')
    parser.add_argument("--unconditional_guidance_scale", type=float, default=7.5, help='scale of unconditional guidance in classifier-free guidance')
    parser.add_argument("--ddim_eta", type=float, default=0.0, help='ddim eta (eta=0.0 corresponds to deterministic sampling')
    parser.add_argument("--is_binary", action='store_true', help='whether binarize sketches or not')
    parser.add_argument("--sample_num", type=int, default=1000000, help='number of samples')
    parser.add_argument("--DDP", action='store_true', help='use DDP, to similar to the training setting')
    parser.add_argument("--world_size", type=int, default=1, help='world size')
    parser.add_argument("--truncation_steps", type=int, default=500, help='early stop the edge condition while sampling')
    parser.add_argument("--return_pred_cond", type=bool, default=True, help='return predicted condition for visualization')
    parser.add_argument('--use_neg_prompt', action='store_true', help='use negative prompt')
 
    return parser

# function to return a path list from a txt file
def get_files_from_txt(path):
    file_list = []
    f = open(path)
    for line in f.readlines():
        line = line.strip("\n")
        file_list.append(line)
        sys.stdout.flush()
    f.close()

    return file_list

# function to compute the gradient tensor of condition adaptor
def cond_fn(x, c, t, cond_configs, blocks_indexes,
            target_cond, model, diffusion_model, 
            criterion, scale=1.0, return_pred_cond=True):
    with torch.enable_grad():
        target_cond = target_cond.requires_grad_(True)
        x = x.requires_grad_(True)
        
        features = diffusion_model.model.diffusion_model.forward_return_features(x=x, timesteps=t, context=c, block_indexes=blocks_indexes)
        
        # upsample features
        upsampled_features = []
        for feat in features:
            feat = F.interpolate(input=feat, size=cond_configs['model_configs']['condition_adaptor_config']['size'], mode='bilinear')
            upsampled_features.append(feat)
        upsampled_features = torch.cat(upsampled_features, dim=1)
        
        # compute the gradient
        x_pred = model(upsampled_features, t)
        loss = criterion(target_cond, x_pred)
        grad = torch.autograd.grad(loss.sum(), x)[0] * scale
        
        if return_pred_cond:
            return grad, x_pred
        else:
            return grad


# main worker for DDP
def main_worker(rank, configs, input_paths):
    options, args, _ = configs['options'], configs['args'], configs['model_configs']
    setup(rank=rank, world_size=args.world_size)
    runner = ConditionAlignerRunner(rank=rank, configs=configs)
    
    if configs['args'].mode == "from_text":
        runner.ddim_sample_with_CA_from_text(input_paths, cond_fn)
    elif configs['args'].mode == "unconditional":
        runner.ddim_sample_with_CA_from_unconditional(input_paths, cond_fn)

if __name__ == "__main__":

    # initialize the configurations
    parser = get_parser()
    args = parser.parse_args()
    setup_seed(args.seed)

    args.inference = True
    os.makedirs(args.outdir, exist_ok=True)
    
    if args.use_neg_prompt:
        args.neg_prompt = DEFAULT_NEGATIVE_PROMPT
    
    # load configurations
    model_configs = OmegaConf.load(args.base)
    configs = {'args': args, 'model_configs': model_configs}

    # load image paths
    input_paths = get_files_from_txt(args.indir)
    print(f"Found {len(input_paths)} inputs.")
    
    # test on multiple GPUs
    if args.DDP:
        mp.spawn(main_worker, nprocs=args.world_size, args=(configs, input_paths))
    # test on single GPU or other devices
    else:
        runner = ConditionAlignerRunner(rank=0, configs=configs)
        
        if configs['args'].mode == "from_text":
            runner.ddim_sample_with_CA_from_text(args, input_paths, cond_fn)
        elif configs['args'].mode == "unconditional":
            args.unconditional_guidance_scale == 1.0
            runner.ddim_sample_with_CA_from_unconditional(args, input_paths, cond_fn)
