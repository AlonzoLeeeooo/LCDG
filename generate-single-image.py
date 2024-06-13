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

device = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_NEGATIVE_PROMPT = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, ' \
                          'fewer digits, cropped, worst quality, low quality'


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
    parser.add_argument("-b", "--base", type=str, default="condition_adaptor_configs/edge.yaml", help="path of config file")
    parser.add_argument("-s", "--seed", type=int, default=23, help="setting seed")
    parser.add_argument("--get_files_from_path", action='store_true', help='get data from folder or flist')
    parser.add_argument("--indir", type=str, default='', help="input images dir")
    parser.add_argument("--caption", type=str, default='', help="text prefix")
    parser.add_argument("--outdir", type=str, default='outputs/sampling', help="output dir")
    parser.add_argument("--resume", type=str, help='checkpoint of condition adaptor')
    parser.add_argument("--resume_from_single_GPU", action='store_true', help='load single-GPU trained model weights')
    parser.add_argument("--resume_from_DDP", action='store_true', help='load DDP trained model weights')
    parser.add_argument("--steps", type=int, default=50, help='number of ddim sampling steps')
    parser.add_argument("--cond_scale", type=float, default=2.0, help='scale of condition score')
    parser.add_argument("--guidance_scale", type=float, default=7.5, help='scale of classifier-free guidance')
    parser.add_argument("--ddim_eta", type=float, default=0.0, help='ddim eta (eta=0.0 corresponds to deterministic sampling')
    parser.add_argument("--is_binary", action='store_true', help='whether binarize sketches or not')
    parser.add_argument("--sample_num", type=int, default=10, help='number of samples')
    parser.add_argument("--add_cond_score", type=bool, default=True, help='add or minus condition score')
    parser.add_argument("--verbose", action='store_true', help='turn on verbose mode')
    parser.add_argument("--DDP", action='store_true', help='use DDP, to similar to the training setting')
    parser.add_argument("--world_size", type=int, default=1, help='world size')
    parser.add_argument("--ddim_sampling", type=bool, default=True, help='use ddim sampling, default by True')
    parser.add_argument("--reverse_cond", action='store_true',
                        help='reverse the value of conditions, default with background in black (meaning that if the background is in while you need to turn this flag on)')
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
            feat = F.interpolate(input=feat, size=cond_configs['size'], mode='bilinear')
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


if __name__ == "__main__":

    # initialize the configurations
    parser = get_parser()
    args = parser.parse_args()
    setup_seed(args.seed)

    if args.use_neg_prompt:
        args.neg_prompt = DEFAULT_NEGATIVE_PROMPT
    args.inference = True
    os.makedirs(args.outdir, exist_ok=True)
    
    
    # load configurations
    model_configs = OmegaConf.load(args.base)
    configs = {'model_configs': model_configs, 'args': args}
        
    if args.DDP:
        assert "We currently only support single-GPU sampling."
    else:
        runner = ConditionAlignerRunner(rank=0, configs=configs)
        
        if model_configs['condition_aligner_config']['mode'] == "from_text":
            runner.T2I_sampling_single_image_with_CA(args, args.indir, cond_fn)
        elif model_configs['condition_aligner_config']['mode'] == "unconditional":
            runner.U2I_sampling_single_image_with_CA(args, args.indir, cond_fn)
