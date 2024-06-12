import os
import cv2
import random
import sys
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from condition_aligner_src.condition_aligner_model import ConditionAligner
from condition_aligner_src.condition_aligner_dataset import ImageTextConditionDataset, ImageConditionDataset

def setup_seed(seed):
    print('Seed: ', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

def get_files(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(os.path.join(root, filespath))
    return ret

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print(f"Setting up the process on rank {rank}.")

def cleanup():
    dist.destroy_process_group()


class ConditionAlignerRunner:
    def __init__(self, rank, configs):
        self.configs = configs
        
        # unpack configs
        self.args, self.model_configs = configs['args'], configs['model_configs']
        self.world_size = self.args.world_size
        
        # initialize device configurations
        self.global_rank = rank
        self.device = f"cuda:{rank}" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.mode = self.model_configs['condition_aligner_config']['mode']
        
        # define tensorboard logger
        # inference        
        self.diffusion_steps = self.model_configs['model']['params']['timesteps']
        self.blocks = self.model_configs['condition_aligner_config']['blocks']
        
        self.diffusion_model = instantiate_from_config(self.model_configs['model'])
        self.diffusion_model.eval().to(self.device)
        
        self.vae_model = self.diffusion_model.first_stage_model
        self.vae_model.to(self.device).eval()
        
        if self.mode == "from_text":
            # define conditioning model of stable diffusion - CLIP
            self.cond_model = self.diffusion_model.cond_stage_model
            self.cond_model.to(self.device).eval()
        
        # define our condition adaptor
        self.model = ConditionAligner(
        time_channels=self.model_configs['condition_aligner_config']['time_channels'],
        in_channels=self.model_configs['condition_aligner_config']['in_channels'],
        out_channels=self.model_configs['condition_aligner_config']['out_channels'],).to(self.device)
        
        self.model.init_weights(init_type='xavier')
        if self.args.DDP:
            self.model = DistributedDataParallel(self.model, device_ids=[self.global_rank], find_unused_parameters=True)
        else:
            self.model.to(self.device)
        
        # resume training from previous checkpoint / load pre-trained checkpoint while inference
        if self.args.resume:
            if self.args.inference:
                # remove ``module.'' from pre-trained checkpoint
                from collections import OrderedDict
                state_dict = torch.load(self.args.resume, map_location='cpu')['model_state_dict']
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    if 'module.' in k:
                        name = k[7:]
                    else:
                        name = k
                    new_state_dict[name] = v
                self.model.load_state_dict(new_state_dict)
                print(f"\nSuccessfully load checkpoint from {self.args.resume}.\n")
            else:
                assert "``args.resume'' should be string!"
        
        # define loss function & optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.model_configs['condition_aligner_config']['learning_rate'])
        
        # define training parameters
        self.iteration = 0
        self.epoch = 0

    def upsample_features(self, size, features):
        upsampled_features = []
        for feat in features:
            feat = nn.functional.interpolate(input=feat, size=size, mode='bilinear')
            upsampled_features.append(feat)
        
        return torch.cat(upsampled_features, dim=1)            
    
    # train from text-to-image diffusion model
    def T2I_model_training(self):
        args = self.args
        CA_configs = self.model_configs['condition_aligner_config']
        
        train_dataset = ImageTextConditionDataset(
            cond_type=CA_configs['cond_type'],
            image_dir=CA_configs['image_dir'],
            cond_dir=CA_configs['cond_dir'],
            text_dir=CA_configs['text_dir'],
            image_size=CA_configs['image_size'],
            kmeans_center=CA_configs['kmeans_center'] if CA_configs['kmeans_center'] is not None else None)
        
        if args.eval_freq > 0 or args.evaluate:
            val_dataset = ImageTextConditionDataset(
                cond_type=CA_configs['cond_type'],
                image_dir=CA_configs['val_image_dir'],
                cond_dir=CA_configs['val_cond_dir'],
                text_dir=CA_configs['val_text_dir'],
                image_size=CA_configs['image_size'],
                kmeans_center=CA_configs['kmeans_center'] if CA_configs['kmeans_center'] is not None else None)
        
        if args.DDP:
            self.train_sampler = DistributedSampler(train_dataset, num_replicas=self.world_size, rank=self.global_rank)
            if args.eval_freq > 0 or args.evaluate:
                self.val_sampler = DistributedSampler(val_dataset, num_replicas=self.world_size, rank=self.global_rank)
        else:
            self.train_sampler = DistributedSampler(train_dataset, num_replicas=1, rank=0)
            if args.eval_freq > 0 or args.evaluate:
                self.val_sampler = DistributedSampler(val_dataset, num_replicas=1, rank=0)
        
        if args.DDP:
            train_loader = DataLoader(train_dataset, shuffle=False, pin_memory=True,
                                      batch_size=args.batch_size // self.world_size,
                                      num_workers=args.num_workers, sampler=self.train_sampler)
            if args.eval_freq > 0 or args.evaluate:
                self.val_loader = DataLoader(val_dataset, shuffle=False, pin_memory=True,
                                      batch_size=1,
                                      num_workers=args.num_workers, sampler=self.val_sampler)
        else:
            train_loader = DataLoader(train_dataset, pin_memory=True,
                                      batch_size=args.batch_size, num_workers=args.num_workers,
                                      sampler=self.train_sampler)
            if args.eval_freq > 0 or args.evaluate:
                self.val_loader = DataLoader(val_dataset, pin_memory=True,
                                      batch_size=1,
                                      num_workers=args.num_workers,
                                      sampler=self.val_sampler)

        print(f"\nCurrent dataloader length: {str(len(train_loader))}.\n")
        self.model.train()
        
        print("\nStart training...\n")
        for epoch in range(args.epochs):
            self.epoch = epoch
            for i, batch in enumerate(train_loader):
                self.iteration += 1
                image, cond, text = batch['image'].to(self.device), batch['cond'].to(self.device), batch['text']
                                
                # foward vector quantatize model & conditioning model (CLIP)
                image = self.vae_model.encode_to_codebook(image)
                cond = self.vae_model.encode_to_codebook(cond)
                
                self.cond_model = self.cond_model.to("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
                text_cond = self.cond_model(text).to(self.device)
                
                # use for saving training samples
                image_sample = image.detach()
                cond_sample = cond.detach()

                # generate noise
                rnd_gen = torch.Generator(device=self.device).manual_seed(args.seed)
                noise = torch.randn(1, CA_configs['out_channels'],
                                    CA_configs['size'], CA_configs['size'],
                                    generator=rnd_gen, device=self.device)

                # forward diffusion model 
                t = torch.randint(0, self.diffusion_steps, (image.size(0),), device=self.device).long()
                noisy_img = self.diffusion_model.q_sample(x_start=image, t=t, noise=noise)
                features = self.diffusion_model.model.diffusion_model.forward_return_features(noisy_img, t, text_cond, block_indexes=self.blocks)

                upsampled_features = self.upsample_features(size=CA_configs['size'], features=features)
                
                self.optimizer.zero_grad()
                
                # forward
                cond_pred = self.model(upsampled_features, t)
                
                # update
                loss = self.criterion(cond, cond_pred)
                
                loss.backward()
                self.optimizer.step()
            
                if self.iteration % args.print_freq == 0:
                    print(f"Iteration: {self.iteration}/{args.max_steps}, Epoch: {epoch+1}/{args.epochs}, " "Loss: %.2f" % loss.item())
                    
                # save model checkpoints
                if self.iteration % args.checkpoint_freq == 0:
                    print("\nSaving checkpoints...\n")
                    os.makedirs(os.path.join(args.logdir, 'checkpoints'), exist_ok=True)
                    save_path = os.path.join(args.logdir, 'checkpoints', f"epoch_{epoch}_iters_{self.iteration}.pth")
                    torch.save({'model_state_dict': self.model.state_dict()}, save_path)
                    print(f"\nSuccessfully save checkpoint epoch_{epoch}_iters_{self.iteration}.pth!\n")
                    
                # save visualization
                if self.iteration % args.sample_freq == 0:
                    print("\nSaving training samples...\n")
                    cond = cond.detach()
                    cond_pred = cond_pred.detach()
                        
                    image_sample = self.vae_model.decode(image_sample)
                    cond_sample = self.vae_model.decode(cond_sample)
                    cond_pred = self.vae_model.decode(cond_pred)
                    
                    item_dict = {}
                    item_dict['image'] = image_sample
                    item_dict['cond'] = cond_sample
                    item_dict['cond_pred'] = cond_pred
                    
                    # visualize
                    self.visualize(item_dict=item_dict, epoch=epoch)
                    
                # make evaluation
                if args.eval_freq > 0 and self.iteration % args.eval_freq == 0:
                    self.evaluate_t2i()
                                
                # kill the process if training reaches maximum steps
                if self.iteration > args.max_steps:
                    save_path = os.path.join(args.logdir, 'checkpoints', f"LAST.pth")
                    os.makedirs(os.path.join(args.logdir, 'checkpoints'), exist_ok=True)
                    torch.save({'model_state_dict': self.model.state_dict()}, save_path)
                    print(f"LAST checkpoint saved.")
                    cleanup()
                    sys.exit(0)
            
            
    # train from unconditional diffusion model such Celeb SD
    def unconditional_model_training(self):
        args = self.args
        CA_configs = self.model_configs['condition_aligner_config']
        
        train_dataset = ImageConditionDataset(
            cond_type=CA_configs['cond_type'],
            image_dir=CA_configs['image_dir'],
            cond_dir=CA_configs['cond_dir'],
            image_size=CA_configs['image_size'])
        
        if args.DDP:
            self.train_sampler = DistributedSampler(train_dataset, num_replicas=self.args.world_size, rank=self.global_rank)
        else:
            self.train_sampler = DistributedSampler(train_dataset, num_replicas=1, rank=0)
        
        if args.DDP:
            train_loader = DataLoader(train_dataset, shuffle=False, pin_memory=True,
                                      batch_size=args.batch_size // self.args.world_size,
                                      num_workers=args.num_workers, sampler=self.train_sampler)
        else:
            train_loader = DataLoader(train_dataset, pin_memory=True,
                                      batch_size=args.batch_size, num_workers=args.num_workers,
                                      sampler=self.train_sampler)

        print(f"\nCurrent dataloader length: {str(len(train_loader))}.\n")
        self.model.train()

        print("\nStart training...\n")
        for epoch in range(args.epochs):
            for i, batch in enumerate(train_loader):
                self.iteration += 1
                image, cond = batch['image'].to(self.device), batch['cond'].to(self.device)
                
                # foward vector quantatize model & conditioning model (CLIP)
                image = self.vae_model.encode_to_codebook(image)
                cond = self.vae_model.encode_to_codebook(cond)
                
                # use for saving training samples
                image_sample = image.detach()
                cond_sample = cond.detach()

                # generate noise
                rnd_gen = torch.Generator(device=f'cuda:{self.global_rank}').manual_seed(args.seed)
                noise = torch.randn(1, CA_configs['out_channels'], CA_configs['size'], CA_configs['size'], generator=rnd_gen, device=f'cuda:{self.global_rank}')

                # forward diffusion model 
                t = torch.randint(0, self.diffusion_steps, (image.size(0),), device=f'cuda:{self.global_rank}').long()
                noisy_img = self.diffusion_model.q_sample(x_start=image, t=t, noise=noise)
                features = self.diffusion_model.model.diffusion_model.forward_return_features(noisy_img, t, block_indexes=self.blocks)

                upsampled_features = self.upsample_features(size=CA_configs['size'], features=features)
                
                self.optimizer.zero_grad()
                
                # forward
                cond_pred = self.model(upsampled_features, t)
                
                # update
                loss = self.criterion(cond, cond_pred)
                loss.backward()
                self.optimizer.step()
                

                if self.iteration % args.print_freq == 0:
                    print(f"Iteration: {self.iteration}/{args.max_steps}, Epoch: {epoch+1}/{args.epochs}, " "Loss: %.2f" % loss.item())
                    
                # save model checkpoints
                if self.iteration % args.checkpoint_freq == 0:
                    print("\nSaving checkpoints...\n")
                    os.makedirs(os.path.join(args.logdir, 'checkpoints'), exist_ok=True)
                    save_path = os.path.join(args.logdir, 'checkpoints', f"epoch_{epoch}_iters_{self.iteration}.pth")
                    torch.save({'model_state_dict': self.model.state_dict()}, save_path)
                    print(f"\nSuccessfully save checkpoint epoch_{epoch}_iters_{self.iteration}.pth!\n")
                    
                # save visualization
                if self.iteration % args.sample_freq == 0:
                    print("\nSaving training samples...\n")
                    cond = cond.detach()
                    cond_pred = cond_pred.detach()
                        
                    image_sample = self.vae_model.decode(image_sample)
                    cond_sample = self.vae_model.decode(cond_sample)
                    cond_pred = self.vae_model.decode(cond_pred)
                    
                    item_dict = {}
                    item_dict['image'] = image_sample
                    item_dict['cond'] = cond_sample
                    item_dict['cond_pred'] = cond_pred
                    
                    # visualize
                    self.visualize(item_dict=item_dict, epoch=epoch)
                                
                # kill the process if training reaches maximum steps
                if self.iteration > args.max_steps:
                    save_path = os.path.join(args.logdir, 'checkpoints', f"LAST_epoch_{epoch}.pth")
                    torch.save({'model_state_dict': self.model.state_dict()}, save_path)
                    cleanup()
                    sys.exit(0)

    # function of saving visualization
    def visualize(self, item_dict, epoch, is_evaluation=False, val_count=0):
        if is_evaluation:
            os.makedirs(os.path.join(self.args.logdir, 'val_samples', f'epoch{self.epoch}_iters_{self.iteration}'), exist_ok=True)
        else:
            os.makedirs(os.path.join(self.args.logdir, 'samples'), exist_ok=True)
        
        concat_dict = {}
        final_dict = {}
        final_final_list = []
        
        for key in item_dict.keys():
            concat_dict[key], final_dict[key] = [], []
            
            item = item_dict[key]
            for i in range(item.size(0)):
                concat_dict[key].append(item[i])
            concat_tensor = torch.cat(concat_dict[key], dim=2)                 # concat on ``width'' dimension
            final_dict[key] = concat_tensor
            
        for key in final_dict.keys():
            item = final_dict[key]                                             # in size of [C, H, N * W]
            final_final_list.append(item)
            
        item = torch.cat(final_final_list, dim=1)                              # concat on ``height'' dimension
            
        item = torch.clamp(((item + 1) * 127.5), min=0.0, max=255.0)
    
        item = item.detach().permute(1, 2, 0).cpu().numpy().astype(np.uint8)        # to size [H, N * W, C]
        item = cv2.cvtColor(item, cv2.COLOR_RGB2BGR)
        
        if is_evaluation:
            cv2.imwrite(os.path.join(self.args.logdir,  'val_samples', f'epoch{self.epoch}_iters_{self.iteration}', f'{val_count}.png'), item)
        else:
            cv2.imwrite(os.path.join(self.args.logdir, 'samples', f'epoch_{epoch}_iters_{self.iteration}.png'), item)


    # save visualization while inferencing edge aligner
    def visualize_while_inference(self, item_dict, filename):
        os.makedirs(os.path.join(self.args.outdir, 'visualization'), exist_ok=True)
        
        concat_dict = {}
        final_dict = {}
        final_final_list = []
        
        for key in item_dict.keys():
            concat_dict[key], final_dict[key] = [], []
            
            item = item_dict[key]
            for i in range(item.size(0)):
                concat_dict[key].append(item[i])
            concat_tensor = torch.cat(concat_dict[key], dim=2)                 # concat on ``width'' dimension
            final_dict[key] = concat_tensor
            
        for key in final_dict.keys():
            item = final_dict[key]                                             # in size of [C, H, N * W]
            final_final_list.append(item)
            
        item = torch.cat(final_final_list, dim=1)                              # concat on ``height'' dimension
            
        item = torch.clamp(((item + 1) * 127.5), min=0.0, max=255.0)
        
        item = item.detach().permute(1, 2, 0).cpu().numpy().astype(np.uint8)        # to size [H, N * W, C]
        item = cv2.cvtColor(item, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(self.args.outdir, 'visualization', f'{filename}.png'), item)
    
    # inference code of condition adaptor, correlates to condition_adaptor_inference.py
    # inference condition adaptor FROM TEXT-TO-IMAGE DIFFUSION MODEL
    def inference_from_text(self, configs):        
        # unpack options and args
        options, args = configs['options'], configs['args']
        
        # DATA: initialize & read in inferece data
        if args.get_files_from_path:
            image_paths = get_files(args.indir)
        else:
            image_paths = get_files_from_txt(args.indir)
        
        # variable for printing progress
        count = 0
        print("\nStart inferencing...\n")
        
        for path in image_paths:
            # kill the process if reach maximum number
            if count > args.inference_num:
                print("\nInferece done...\n")
                sys.exit(0)
                
            count += 1
            print(f"Progress: {count}/{len(image_paths)}")
            
            image = cv2.imread(path)
            filename = os.path.basename(path).split('.')[0]
            subfolder = path.split('/')[-2]
            
            # read in image
            image = cv2.resize(image, (options['image_size'], options['image_size']))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = torch.from_numpy(image.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0)
            
            # read in ground truth condition
            cond = cv2.imread(os.path.join(args.cond, filename + '.png'))
            # cond = cv2.imread(os.path.join(args.cond, subfolder, filename + '.jpg'))
            cond = cv2.resize(cond, (options['image_size'], options['image_size']))
            if self.options['cond_type'] == "edge":
                _, cond = cv2.threshold(cond, thresh=180.0, maxval=255.0, type=cv2.THRESH_BINARY)
            if self.options['cond_type'] == "saliency":
                _, cond = cv2.threshold(cond, thresh=127.5, maxval=255.0, type=cv2.THRESH_BINARY)
            cond = torch.from_numpy(cond.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0)
            
            # read in text
            # with open(os.path.join(args.text, subfolder, filename + '.txt')) as f:
            #     for line in f.readlines():
            #         text = line
            # f.close()
            
            # FORWARDING
            image = image.to(self.device)
            cond = cond.to(self.device)
            
            # forward VQ model
            image = self.vae_model.encode_to_codebook(image)
            
            # forward pre-trained CLIP
            self.cond_model = self.cond_model.to(self.device)
            text_cond = self.cond_model(['']).to(self.device)
            
            # generate noise
            rnd_gen = torch.Generator(device=f'cuda:{self.global_rank}').manual_seed(args.seed)
            noise = torch.randn(1, options['out_channels'], options['size'], options['size'], generator=rnd_gen, device=f'cuda:{self.global_rank}')
            
            # forward diffusion model, obtain the deep features            
            t = torch.randint(0, self.diffusion_steps, (image.size(0),), device=f'cuda:{self.global_rank}').long()
            noisy_img = self.diffusion_model.q_sample(x_start=image, t=t, noise=noise)
            features = self.diffusion_model.model.diffusion_model.forward_return_features(noisy_img, t, text_cond, block_indexes=self.blocks)
            
            # align the feature via upsampling
            upsampled_features = self.upsample_features(size=options['size'], features=features)
            
            # forward our pre-trained edge aligner
            cond_pred = self.model(upsampled_features, t)
                
            image_sample = self.vae_model.decode(image)
            cond_pred = self.vae_model.decode(cond_pred)
            
            item_dict = {}
            item_dict['image'] = image_sample
            item_dict['cond'] = cond
            item_dict['cond_pred'] = cond_pred
            
            self.visualize_while_inference(item_dict=item_dict, filename=filename)
            
    # inference code of condition adaptor FROM UNCONDITIONAL DIFFUSION MODEL
    def inference_from_unconditional(self, configs):        
        # unpack options and args
        options, args = configs['options'], configs['args']
        
        # DATA: initialize & read in inferece data
        if args.get_files_from_path:
            image_paths = get_files(args.indir)
        else:
            image_paths = get_files_from_txt(args.indir)
        
        # variable for printing progress
        count = 0
        print("\nStart inferencing...\n")
        
        for path in image_paths:
            # kill the process if reach maximum number
            if count > args.inference_num:
                print("\nInferece done...\n")
                sys.exit(0)
                
            count += 1
            print(f"Progress: {count}/{len(image_paths)}")
            
            image = cv2.imread(path)
            filename = os.path.basename(path).split('.')[0]
            
            # read in image
            image = cv2.resize(image, (options['image_size'], options['image_size']))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = torch.from_numpy(image.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0)
            
            # read in ground truth condition
            cond = cv2.imread(os.path.join(args.cond, filename + '.png'))
            
            cond = cv2.resize(cond, (options['image_size'], options['image_size']))
            if self.options['cond_type'] == "edge":
                _, cond = cv2.threshold(cond, thresh=180.0, maxval=255.0, type=cv2.THRESH_BINARY)
            if self.options['cond_type'] == "saliency":
                _, cond = cv2.threshold(cond, thresh=127.5, maxval=255.0, type=cv2.THRESH_BINARY)
            cond = torch.from_numpy(cond.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0)
            
            # FORWARDING
            image = image.to(self.device)
            cond = cond.to(self.device)
            
            # forward VQ model
            image = self.vae_model.encode_to_codebook(image)
            
            # generate noise
            rnd_gen = torch.Generator(device=f'cuda:{self.global_rank}').manual_seed(args.seed)
            noise = torch.randn(1, options['out_channels'], options['size'], options['size'], generator=rnd_gen, device=f'cuda:{self.global_rank}')
            
            # forward diffusion model, obtain the deep features            
            t = torch.randint(0, self.diffusion_steps, (image.size(0),), device=f'cuda:{self.global_rank}').long()
            noisy_img = self.diffusion_model.q_sample(x_start=image, t=t, noise=noise)
            noisy_img_vis = noisy_img.detach()
            features = self.diffusion_model.model.diffusion_model.forward_return_features(noisy_img, t, block_indexes=self.blocks)
            
            # align the feature via upsampling
            upsampled_features = self.upsample_features(size=options['size'], features=features)
            
            # forward our pre-trained edge aligner
            cond_pred = self.model(upsampled_features, t)
                
            image_sample = self.vae_model.decode(image)
            cond_pred = self.vae_model.decode(cond_pred)
            noisy_img_vis = self.vae_model.decode(noisy_img_vis)
            
            item_dict = {}
            item_dict['image'] = image_sample
            item_dict['noisy_img'] = noisy_img_vis
            item_dict['cond'] = cond
            item_dict['cond_pred'] = cond_pred
            
            self.visualize_while_inference(item_dict=item_dict, filename=filename)
    
    
    # save visualization while sampling
    def visualize_while_sampling(self, args, item_dict, filename):
        
        # create output dir if not exist
        os.makedirs(self.args.outdir, exist_ok=True)
        
        concat_dict = {}
        final_dict = {}
        final_final_list = []
        
        for key in item_dict.keys():
            concat_dict[key], final_dict[key] = [], []
            
            item = item_dict[key]
            for i in range(item.size(0)):
                concat_dict[key].append(item[i])
            concat_tensor = torch.cat(concat_dict[key], dim=2)                 # concat on ``width'' dimension
            final_dict[key] = concat_tensor
            
        for key in final_dict.keys():
            item = final_dict[key]                                             # in size of [C, H, N * W]
            final_final_list.append(item)
            
        item = torch.cat(final_final_list, dim=1)                              # concat on ``height'' dimension
            
        item = torch.clamp(((item + 1) * 127.5), min=0.0, max=255.0)
        
        item = item.detach().permute(1, 2, 0).cpu().numpy().astype(np.uint8)        # to size [H, N * W, C]
        item = cv2.cvtColor(item, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(os.path.join(self.args.outdir, f'{filename}.png'), item)
    
    
    
    # sample from the pre-trained diffusion model with condition adaptor FROM TEXT-TO-IMAGE DIFFUSION MODELS (CA)
    def ddim_sample_with_CA_from_text(self, args, input_paths, cond_fn):
        # initialize condition adapter (CA) configs
        CA_configs = self.model_configs['condition_aligner_config']

        # initialize the models
        model = self.model
        diffusion_model = self.diffusion_model
        sampler = DDIMSampler(diffusion_model)
        vae_model = self.vae_model
        cond_stage_model = self.cond_model
        
        # initialize criterion for condition adaptor
        criterion = nn.MSELoss()
        
        # start evaluation
        with torch.no_grad():
            with diffusion_model.ema_scope():
                count = 0
                print("\nStart sampling...\n")
                for path in tqdm(input_paths):
                    
                    filename = os.path.basename(path).split('.')[0]
                    
                    # LOAD IN THE DATA
                    # read in image - for visualization
                    image = cv2.imread(path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, (CA_configs['image_size'], CA_configs['image_size']))
                    image = torch.from_numpy(image.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
                    
                    # read in target condition
                    target_cond = cv2.imread(os.path.join(self.args.target_cond, filename + '.png'))    
                    target_cond = cv2.resize(target_cond, (CA_configs['image_size'], CA_configs['image_size']))
                    
                    if CA_configs['cond_type'] == 'edge':
                        _, target_cond = cv2.threshold(target_cond, 180, 255.0, cv2.THRESH_BINARY)
                    if CA_configs['cond_type'] == 'saliency':
                        _, target_cond = cv2.threshold(target_cond, 127.5, 255.0, cv2.THRESH_BINARY)
                        
                    if self.args.reverse_cond:
                        target_cond = 255.0 - target_cond
                        target_cond = torch.from_numpy(target_cond.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
                    else:
                        target_cond = torch.from_numpy(target_cond.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
                    
                    target_cond_vis = target_cond.detach()

                    # read in correpsonding text
                    with open(os.path.join(self.args.text, filename + '.txt')) as f:
                        for line in f.readlines():
                            text = line
                        f.close()
                    
                    target_cond = vae_model.encode_to_codebook(target_cond)                
                    
                    # pass text forward condition model - CLIP
                    text_cond = cond_stage_model(text)                
                        
                    # classifier-free guidance
                    if self.args.use_neg_prompt:
                        uc = diffusion_model.get_learned_conditioning(text_cond.shape[0] * [self.args.neg_prompt])
                    else:
                        uc = diffusion_model.get_learned_conditioning(text_cond.shape[0] * [""])

                    # shape of Gaussian noise
                    shape = (CA_configs['out_channels'], CA_configs['size'], CA_configs['size'])
                    outputs = sampler.sample(S=self.args.steps,
                                            conditioning=text_cond,
                                            batch_size=text_cond.shape[0], 
                                            shape=shape,
                                            verbose=False,
                                            unconditional_guidance_scale=self.args.unconditional_guidance_scale,     
                                            unconditional_conditioning=uc,              
                                            eta=self.args.ddim_eta,
                                            target_cond=target_cond,
                                            cond_fn=cond_fn,
                                            cond_model=model,
                                            blocks_indexes=CA_configs['blocks'],
                                            cond_configs=self.configs,
                                            cond_criterion=criterion,
                                            cond_scale=self.args.cond_scale,
                                            add_cond_score=self.args.add_cond_score,
                                            truncation_steps=self.args.truncation_steps,
                                            return_pred_cond=self.args.return_pred_cond)
                    
                    # unpack returned outputs
                    if self.args.return_pred_cond:
                        samples_ddim, _, pred_cond = outputs
                    else:
                        samples_ddim, _ = outputs
                        
                    x_samples_ddim = diffusion_model.decode_first_stage(samples_ddim)
                    if self.args.return_pred_cond:
                        pred_cond = diffusion_model.decode_first_stage(pred_cond)

                    item_dict = {}
                    item_dict['image'] = image
                    item_dict['target_cond'] = target_cond_vis
                    if self.args.return_pred_cond:
                        item_dict['pred_cond'] = pred_cond
                    item_dict['sample'] = x_samples_ddim
                    
                    # save visualizations
                    self.visualize_while_sampling(args, item_dict, filename)

                    # stop the code if reach maximum number
                    count += 1 
                    if count >= self.args.sample_num:
                        print("\nStop sampling...\n")
                        cleanup()
                        sys.exit(0)
                        
    # sample from the pre-trained diffusion model with condition adaptor FROM UNCONDITIONAL DIFFUSION MODELS (CA)
    def ddim_sample_with_CA_from_unconditional(self, args, input_paths, cond_fn):
        
        # initialize the models
        model = self.model
        diffusion_model = self.diffusion_model
        sampler = DDIMSampler(diffusion_model)
        vae_model = self.vae_model
        
        # initialize criterion for condition adaptor
        criterion = nn.MSELoss()
        
        # start evaluation
        with torch.no_grad():
            with diffusion_model.ema_scope():
                count = 0
                print("\nStart sampling...\n")
                for path in tqdm(input_paths):
                    
                    filename = os.path.basename(path).split('.')[0]
                    subfolder = path.split('/')[-2]
                    
                    # LOAD IN THE DATA
                    # read in image - for visualization
                    image = cv2.imread(path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, (self.options['image_size'], self.options['image_size']))
                    image = torch.from_numpy(image.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
                    
                    # read in target condition
                    # cond_paths = get_files(self.args.target_cond)
            
                    # read in cond from path OR read in formulated dataset
                    # target_cond = cv2.imread(cond_paths[count])
                    target_cond = cv2.imread(os.path.join(self.args.target_cond, filename + '.png'))    
                    
                    target_cond = cv2.resize(target_cond, (self.options['image_size'], self.options['image_size']))
                    
                    if self.options['cond_type'] == 'stroke' or 'image':
                        target_cond = cv2.cvtColor(target_cond, cv2.COLOR_BGR2RGB)
                    if self.options['cond_type'] == 'edge':
                        _, target_cond = cv2.threshold(target_cond, 180, 255.0, cv2.THRESH_BINARY)
                    if self.options['cond_type'] == 'saliency':
                        _, target_cond = cv2.threshold(target_cond, 127.5, 255.0, cv2.THRESH_BINARY)
                        
                    if self.args.reverse_cond:
                        target_cond = 255.0 - target_cond
                        target_cond = torch.from_numpy(target_cond.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
                    else:
                        target_cond = torch.from_numpy(target_cond.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
                    
                    target_cond_vis = target_cond.detach()
                    
                    target_cond = vae_model.encode_to_codebook(target_cond)                              
                    
                    # shape of Gaussian noise
                    shape = (3, self.options['size'], self.options['size'])
                    outputs = sampler.sample(S=self.args.steps,
                                            conditioning=None,
                                            batch_size=1, 
                                            shape=shape,
                                            verbose=False,
                                            unconditional_guidance_scale=self.args.unconditional_guidance_scale,     
                                            unconditional_conditioning=None,              
                                            eta=self.args.ddim_eta,
                                            target_cond=target_cond,
                                            cond_fn=cond_fn,
                                            cond_model=model,
                                            blocks_indexes=self.options['blocks'],
                                            cond_configs=self.configs,
                                            cond_criterion=criterion,
                                            cond_scale=self.args.cond_scale,
                                            add_cond_score=self.args.add_cond_score,
                                            truncation_steps=self.args.truncation_steps,
                                            return_pred_cond=self.args.return_pred_cond)
                    
                    # unpack returned outputs
                    if self.args.return_pred_cond:
                        samples_ddim, _, pred_cond = outputs
                    else:
                        samples_ddim, _ = outputs
                        
                    x_samples_ddim = diffusion_model.decode_first_stage(samples_ddim)
                    if self.args.return_pred_cond:
                        pred_cond = diffusion_model.decode_first_stage(pred_cond)

                    item_dict = {}
                    item_dict['image'] = image
                    item_dict['target_cond'] = target_cond_vis
                    if self.args.return_pred_cond:
                        item_dict['pred_cond'] = pred_cond
                    item_dict['sample'] = x_samples_ddim
                    
                    # save visualizations
                    self.visualize_while_sampling(args, item_dict, filename)

                    # stop the code if reach maximum number
                    count += 1 
                    if count >= self.args.sample_num:
                        print("\nStop sampling...\n")
                        cleanup()
                        sys.exit(0)

    # sample from the pre-trained diffusion model with condition adaptor FROM TEXT-TO-IMAGE DIFFUSION MODELS (CA)
    def T2I_sampling_single_image_with_CA(self, args, input_path, cond_fn):
        CA_configs = self.model_configs['condition_aligner_config']
        
        # initialize the models
        model = self.model
        diffusion_model = self.diffusion_model
        sampler = DDIMSampler(diffusion_model)
        vae_model = self.vae_model
        cond_stage_model = self.cond_model
        
        # initialize criterion for condition adaptor
        criterion = nn.MSELoss()
        
        # start evaluation
        with torch.no_grad():
            with diffusion_model.ema_scope():
                count = 0
                print("\nStart sampling...\n")
                    
                filename = os.path.basename(input_path).split('.')[0]
                
                # LOAD IN THE DATA
                # read in target condition
                target_cond = cv2.imread(input_path)    
                target_cond = cv2.resize(target_cond, (CA_configs['image_size'], CA_configs['image_size']))
                
                if CA_configs['cond_type'] == 'stroke' or CA_configs['cond_type'] == 'image':
                    target_cond = cv2.cvtColor(target_cond, cv2.COLOR_RGB2BGR)
                if CA_configs['cond_type'] == 'edge':
                    _, target_cond = cv2.threshold(target_cond, 127.5, 255.0, cv2.THRESH_BINARY)
                if CA_configs['cond_type'] == 'saliency':
                    _, target_cond = cv2.threshold(target_cond, 127.5, 255.0, cv2.THRESH_BINARY)
                    
                if self.args.reverse_cond:
                    target_cond = 255.0 - target_cond
                    target_cond = torch.from_numpy(target_cond.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
                else:
                    target_cond = torch.from_numpy(target_cond.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
                
                target_cond_vis = target_cond.detach()
                
                target_cond = vae_model.encode_to_codebook(target_cond)                
                
                # pass text forward condition model - CLIP
                text_cond = cond_stage_model(self.args.caption)                
                    
                # classifier-free guidance
                if self.args.unconditional_guidance_scale != 1.0:
                    if self.args.use_neg_prompt:
                        uc = diffusion_model.get_learned_conditioning(text_cond.shape[0] * [self.args.neg_prompt])
                    else:
                        uc = diffusion_model.get_learned_conditioning(text_cond.shape[0] * [""])

                # shape of Gaussian noise
                shape = (CA_configs['out_channels'], CA_configs['size'], CA_configs['size'])
                
                outputs = sampler.sample(S=self.args.steps,
                                         conditioning=text_cond,
                                         batch_size=text_cond.shape[0], 
                                         shape=shape,
                                         verbose=False,
                                         unconditional_guidance_scale=self.args.unconditional_guidance_scale,     
                                         unconditional_conditioning=uc,              
                                         eta=self.args.ddim_eta,
                                         target_cond=target_cond,
                                         cond_fn=cond_fn,
                                         cond_model=model,
                                         blocks_indexes=CA_configs['blocks'],
                                         cond_configs=CA_configs,
                                         cond_criterion=criterion,
                                         cond_scale=self.args.cond_scale,
                                         add_cond_score=self.args.add_cond_score,
                                         truncation_steps=self.args.truncation_steps,
                                         return_pred_cond=self.args.return_pred_cond)
                
                # unpack returned outputs
                if self.args.return_pred_cond:
                    samples_ddim, _, pred_cond = outputs
                else:
                    samples_ddim, _ = outputs
                    
                x_samples_ddim = diffusion_model.decode_first_stage(samples_ddim)
                if self.args.return_pred_cond:
                    pred_cond = diffusion_model.decode_first_stage(pred_cond)

                item_dict = {}
                if self.args.return_pred_cond:
                    item_dict['pred_cond'] = pred_cond
                item_dict['sample'] = x_samples_ddim
                
                # save visualizations
                self.visualize_while_sampling(args, item_dict, f"{filename}")

                # stop the code if reach maximum number
                count += 1 
                if count >= self.args.sample_num:
                    print("\nStop sampling...\n")
                    cleanup()
                    sys.exit(0)

    # sample SINGLE IMAGE with condition adaptor FROM UNCONDITIONAL DIFFUSION MODELS (CA)
    def U2I_sampling_single_image_with_CA(self, args, input_path, cond_fn):
        CA_configs = self.model_configs['condition_aligner_config']
        
        # initialize the models
        model = self.model
        diffusion_model = self.diffusion_model
        sampler = DDIMSampler(diffusion_model)
        vae_model = self.vae_model
        
        # initialize criterion for condition adaptor
        criterion = nn.MSELoss()
        
        # start evaluation
        with torch.no_grad():
            with diffusion_model.ema_scope():
                count = 0
                print("\nStart sampling...\n")
                    
                filename = os.path.basename(input_path).split('.')[0]
        
                # read in cond from path OR read in formulated dataset
                target_cond = cv2.imread(input_path)    
                target_cond = cv2.resize(target_cond, (CA_configs['image_size'], CA_configs['image_size']))
                
                if CA_configs['cond_type'] == 'stroke' or 'image':
                    target_cond = cv2.cvtColor(target_cond, cv2.COLOR_BGR2RGB)
                if CA_configs['cond_type'] == 'edge':
                    _, target_cond = cv2.threshold(target_cond, 180, 255.0, cv2.THRESH_BINARY)
                if CA_configs['cond_type'] == 'saliency':
                    _, target_cond = cv2.threshold(target_cond, 127.5, 255.0, cv2.THRESH_BINARY)
                    
                target_cond = torch.from_numpy(target_cond.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
                
                target_cond_vis = target_cond.detach()
                
                target_cond = vae_model.encode_to_codebook(target_cond)                              

                # shape of Gaussian noise
                shape = (3, CA_configs['size'], CA_configs['size'])
                outputs = sampler.sample(S=self.args.steps,
                                         conditioning=None,
                                         batch_size=1, 
                                         shape=shape,
                                         verbose=False,
                                         unconditional_guidance_scale=self.args.unconditional_guidance_scale,     
                                         unconditional_conditioning=None,              
                                         eta=self.args.ddim_eta,
                                         target_cond=target_cond,
                                         cond_fn=cond_fn,
                                         cond_model=model,
                                         blocks_indexes=CA_configs['blocks'],
                                         cond_configs=CA_configs,
                                         cond_criterion=criterion,
                                         cond_scale=self.args.cond_scale,
                                         add_cond_score=self.args.add_cond_score,
                                         truncation_steps=self.args.truncation_steps,
                                         return_pred_cond=self.args.return_pred_cond)
                
                # unpack returned outputs
                if self.args.return_pred_cond:
                    samples_ddim, _, pred_cond = outputs
                else:
                    samples_ddim, _ = outputs
                    
                x_samples_ddim = diffusion_model.decode_first_stage(samples_ddim)
                if self.args.return_pred_cond:
                    pred_cond = diffusion_model.decode_first_stage(pred_cond)

                item_dict = {}
                    
                if self.args.return_pred_cond:
                    item_dict['pred_cond'] = pred_cond
                item_dict['sample'] = x_samples_ddim
                
                # save visualizations
                self.visualize_while_sampling(args, item_dict, filename)

                # stop the code if reach maximum number
                count += 1 
                if count >= self.args.sample_num:
                    print("\nStop sampling...\n")
                    cleanup()
                    sys.exit(0)

