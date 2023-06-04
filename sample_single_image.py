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

from condition_adaptor_src import ConditionAdaptorRunner

device = "cuda" if torch.cuda.is_available() else "cpu"

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
    parser.add_argument("--cond_type", type=str, default='edge', help='type of condition: edge, stroke or saliency')
    parser.add_argument("-b", "--base", type=str, default="condition_adaptor_configs/edge_baseline", help="path of config file")
    parser.add_argument("-s", "--seed", type=int, default=23, help="setting seed")
    parser.add_argument("--indir", type=str, default='', help="input images dir")
    parser.add_argument("--caption", type=str, default='', help="text prefix")
    parser.add_argument("--outdir", type=str, default='workdir/verbose_sampling', help="output dir")
    parser.add_argument("--resume", type=str, help='checkpoint of condition adaptor')
    parser.add_argument("--steps", type=int, default=50, help='number of ddim sampling steps')
    parser.add_argument("--cond_scale", type=float, default=2.0, help='scale of condition score')
    parser.add_argument("--unconditional_guidance_scale", type=float, default=9.0, help='scale of unconditional guidance in classifier-free guidance')
    parser.add_argument("--ddim_eta", type=float, default=0.0, help='ddim eta (eta=0.0 corresponds to deterministic sampling')
    parser.add_argument("--verbose", action='store_true', help='turn on verbose mode')
    parser.add_argument("--DDP", action='store_true', help='use DDP, to similar to the training setting')
    parser.add_argument("--world_size", type=int, default=1, help='world size')
    parser.add_argument("--ddim_sampling", type=bool, default=True, help='use ddim sampling, default by True')
    parser.add_argument("--truncation_steps", type=int, default=500, help='early stop the edge condition while sampling')
    
    # TODO: code to be tuned
    parser.add_argument('--saliency_threshold', default=127.5, type=int, help='threshold of saliency condition for binarizing')
    parser.add_argument('--added_prompt', default="best quality, extremely detailed", help='active prompts')
    parser.add_argument('--negative_prompt', default="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, ugly, boring, bad anatomy, blurry, pixelated, trees, green, obscure, unnatural colors, poor lighting, dull, and unclear",
                        help="negative prompt, other options: ``ugly, boring, bad anatomy, blurry, pixelated, trees, green, obscure, unnatural colors, poor lighting, dull, and unclear''")
    parser.add_argument('--height', default=512, type=int, help='image height')
    parser.add_argument('--width', default=512, type=int, help='image width')
    parser.add_argument('--channels', default=4, type=int, help='channels of the codebook')
    parser.add_argument('--downsampled_factor', default=8, type=int, help='downsampled factor of the vector quantized model')
    
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

# TODO: function to compute the gradient tensor of condition adaptor
def cond_fn(x, c, t, cond_configs, blocks_indexes,
            target_cond, model, diffusion_model, 
            criterion, scale=1.0):
    with torch.enable_grad():
        target_cond = target_cond.requires_grad_(True)
        x = x.requires_grad_(True)
        
        features = diffusion_model.model.diffusion_model.forward_return_features(x=x, timesteps=t, context=c, block_indexes=blocks_indexes)
        
        # upsample features
        upsampled_features = []
        for feat in features:
            feat = F.interpolate(input=feat, size=cond_configs['size'], mode='bilinear')
            upsampled_features.append(feat)
        upsampled_features = torch.cat(upsampled_features, dim=1)
        
        # compute the gradient
        x_pred = model(upsampled_features, t)
        
        # compute loss value
        loss = criterion(target_cond, x_pred)
        
        grad = torch.autograd.grad(loss, x, allow_unused=True)[0] * scale      # original: loss.sum()
        
        return grad

if __name__ == "__main__":

    # TODO: initialize the configurations
    parser = get_parser()
    args = parser.parse_args()
    setup_seed(args.seed)

    args.inference = True
    os.makedirs(args.outdir, exist_ok=True)
    
    # load configurations
    model_configs = OmegaConf.load(args.base)
    configs = {'model_configs': model_configs, 'args': args}
    
    if args.DDP:
        assert "We currently only support single-GPU sampling."
    else:
        runner = ConditionAdaptorRunner(rank=0, configs=configs)
        
        if model_configs['condition_adaptor_config']['mode'] == "from_text":
            runner.T2I_sampling_single_image_with_CA(args, args.indir, cond_fn)
        elif model_configs['condition_adaptor_config']['mode'] == "unconditional":
            runner.U2I_sampling_single_image_with_CA(args, args.indir, cond_fn)
