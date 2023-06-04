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
from condition_adaptor_src.condition_adaptor_model import ConditionAdaptor, StyleConditionAdaptor, \
                                                          ConditionAdaptor1x1Conv, ConditionAdaptorSmall, ConditionAdaptorTiny
from condition_adaptor_src.condition_adaptor_dataset import T2ICollectedDataset, U2ICollectedDataset, U2IInpaintingDataset, T2IInpaintingDataset
from condition_adaptor_src.condition_adaptor_loss import StyleLoss, CLIPStyleLoss

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


class ConditionAdaptorRunner:
    def __init__(self, rank, configs):
        self.configs = configs
        
        # TODO: unpack configs
        self.args, self.model_configs = configs['args'], configs['model_configs']
        self.world_size = self.args.world_size
        
        self.global_rank = rank
        self.mode = self.model_configs['condition_adaptor_config']['mode']
        
        # TODO: define tensorboard logger
        # inference
        if self.args.inference:
            os.makedirs(self.args.outdir, exist_ok=True)
        # train
        else:
            os.makedirs(os.path.join(self.args.logdir, "tensorboard_logs"), exist_ok=True)
            self.logger = SummaryWriter(log_dir=os.path.join(self.args.logdir, "tensorboard_logs"))
        
        self.diffusion_steps = self.model_configs['model']['params']['timesteps']
        self.blocks = self.model_configs['condition_adaptor_config']['blocks']
        
        self.diffusion_model = instantiate_from_config(self.model_configs['model'])
        self.diffusion_model.eval().cuda(self.global_rank)
        
        self.vq_model = self.diffusion_model.first_stage_model
        self.vq_model.cuda(self.global_rank).eval()
        
        if self.mode == "from_text" or self.mode == "t2i_inpainting":
            # TODO: define conditioning model of stable diffusion - CLIP
            self.cond_model = self.diffusion_model.cond_stage_model
            self.cond_model.cuda(self.global_rank).eval()
        
        # TODO: define our condition adaptor
        self.model = ConditionAdaptor(
            time_channels=self.model_configs['condition_adaptor_config']['time_channels'],
            in_channels=self.model_configs['condition_adaptor_config']['in_channels'],
            out_channels=self.model_configs['condition_adaptor_config']['out_channels'],).cuda(self.global_rank)
            
        self.model.init_weights(init_type='xavier')
        if self.args.DDP:
            self.model = DistributedDataParallel(self.model, device_ids=[self.global_rank], find_unused_parameters=True)
        else:
            self.model.cuda(self.global_rank)
        
        # TODO: load pre-trained checkpoint while inference
        if self.args.resume:
            if self.args.inference:
                # TODO: remove ``module.'' from pre-trained checkpoint
                if self.args.resume_from_DDP:
                    from collections import OrderedDict
                    state_dict = torch.load(self.args.resume, map_location='cpu')['model_state_dict']
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:]
                        new_state_dict[name] = v
                    self.model.load_state_dict(new_state_dict)
                else:
                    self.model.load_state_dict(torch.load(self.args.resume, map_location='cpu')['model_state_dict'])
                print(f"\nSuccessfully load checkpoint from {self.args.resume}.\n")
            else:
                assert "``args.resume'' should be string!"
        
        # TODO: define loss function & optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.model_configs['condition_adaptor_config']['learning_rate'])
        
        # TODO: define training parameters
        self.iteration = 0
        self.epoch = 0

    def upsample_features(self, size, features):
        upsampled_features = []
        for feat in features:
            feat = nn.functional.interpolate(input=feat, size=size, mode='bilinear')
            upsampled_features.append(feat)
        
        return torch.cat(upsampled_features, dim=1)

    # TODO: evaluation
    def evaluate_t2i(self):
        args = self.args
        print(f'\nStart evaluation of iteration {self.iteration} and epoch {self.epoch}...\n')
        with torch.no_grad():
            loss = 0.0
            val_count = 0
            for i, batch in enumerate(self.val_loader):
                image, cond, text = batch['image'].cuda(self.global_rank), batch['cond'].cuda(self.global_rank), batch['text']
                
                # TODO: foward vector quantatize model & text condition model (CLIP)
                image = self.vq_model.encode_to_codebook(image)
                cond = self.vq_model.encode_to_codebook(cond)
                
                self.cond_model = self.cond_model.cuda(0)
                text_cond = self.cond_model(text).cuda(self.global_rank)
                
                # use for saving training samples
                image_sample = image.detach()
                cond_sample = cond.detach()

                # TODO: add t-step noise
                rnd_gen = torch.Generator(device=f'cuda:{self.global_rank}').manual_seed(self.args.seed)
                noise = torch.randn(1, self.model_configs['condition_adaptor_config']['out_channels'],
                                    self.model_configs['condition_adaptor_config']['size'], self.model_configs['condition_adaptor_config']['size'],
                                    generator=rnd_gen, device=f'cuda:{self.global_rank}')

                # TODO: forward diffusion model
                # TODO: we focus on the prediction within the trucation steps
                # t_int should be formualted as ``torch.tensor[int(int_num)]''
                t_int = [int(self.model_configs['condition_adaptor_config']['val_diffusion_steps'][0] + \
                    (self.model_configs['condition_adaptor_config']['val_diffusion_steps'][1] - self.model_configs['condition_adaptor_config']['val_diffusion_steps'][0]) * \
                    (i / len(self.val_loader)))]
                t = torch.tensor(t_int).cuda(self.global_rank).long()
    
                
                noisy_img = self.diffusion_model.q_sample(x_start=image, t=t, noise=noise)
                features = self.diffusion_model.model.diffusion_model.forward_return_features(noisy_img, t, text_cond, block_indexes=self.blocks)

                upsampled_features = self.upsample_features(size=self.model_configs['condition_adaptor_config']['size'], features=features)
                
                self.optimizer.zero_grad()
                
                # forward
                cond_pred = self.model(upsampled_features, t)
                
                # update
                loss += self.criterion(cond, cond_pred)
                
                # TODO: the larger the count is, the noisier the sample is
                val_count += 1    
                # TODO: save visualization
                if i % self.model_configs['condition_adaptor_config']['val_sample_freq'] == 0:
                    print(f"Progress: {i}")
                    cond = cond.detach()
                    cond_pred = cond_pred.detach()
                        
                    image_sample = self.vq_model.decode(image_sample)
                    cond_sample = self.vq_model.decode(cond_sample)
                    cond_pred = self.vq_model.decode(cond_pred)
                    
                    item_dict = {}
                    item_dict['image'] = image_sample
                    item_dict['cond'] = cond_sample
                    item_dict['cond_pred'] = cond_pred
                    
                    # visualize
                    self.visualize(item_dict=item_dict, epoch=self.epoch, is_evaluation=True, val_count=val_count)
                 
            # TODO: calculate average loss and record it   
            loss /= len(self.val_loader)
            self.logger.add_scalar('avg_loss/eval', loss.item(), self.iteration)
            print("\nEvaluation done.\n")
    
    # TODO: train from text-to-image diffusion model
    def T2I_training(self):
        args = self.args
        CA_configs = self.model_configs['condition_adaptor_config']
        
        train_dataset = T2ICollectedDataset(
            cond_type=CA_configs['cond_type'],
            image_dir=CA_configs['image_dir'],
            cond_dir=CA_configs['cond_dir'],
            text_dir=CA_configs['text_dir'],
            image_size=CA_configs['image_size'],
            kmeans_center=CA_configs['kmeans_center'])
        
        if args.eval_freq > 0:
            val_dataset = T2ICollectedDataset(
                cond_type=CA_configs['cond_type'],
                image_dir=CA_configs['val_image_dir'],
                cond_dir=CA_configs['val_cond_dir'],
                text_dir=CA_configs['val_text_dir'],
                image_size=CA_configs['image_size'],
                kmeans_center=CA_configs['kmeans_center'])
            
        if args.DDP:
            self.train_sampler = DistributedSampler(train_dataset, num_replicas=self.world_size, rank=self.global_rank)
            if args.eval_freq > 0:
                self.val_sampler = DistributedSampler(val_dataset, num_replicas=self.world_size, rank=self.global_rank)
        else:
            self.train_sampler = DistributedSampler(train_dataset, num_replicas=1, rank=0)
            if args.eval_freq > 0:
                self.val_sampler = DistributedSampler(val_dataset, num_replicas=1, rank=0)
        
        if args.DDP:
            train_loader = DataLoader(train_dataset, shuffle=False, pin_memory=True,
                                      batch_size=args.batch_size // self.world_size,
                                      num_workers=args.num_workers, sampler=self.train_sampler)
            if args.eval_freq > 0:
                self.val_loader = DataLoader(val_dataset, shuffle=False, pin_memory=True,
                                      batch_size=1,
                                      num_workers=args.num_workers, sampler=self.val_sampler)
        else:
            train_loader = DataLoader(train_dataset, pin_memory=True,
                                      batch_size=args.batch_size, num_workers=args.num_workers,
                                      sampler=self.train_sampler)
            if args.eval_freq > 0:
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
                image, cond, text = batch['image'].cuda(self.global_rank), batch['cond'].cuda(self.global_rank), batch['text']
                
                # TODO: foward vector quantatize model & text condition model (CLIP)
                image = self.vq_model.encode_to_codebook(image)
                cond = self.vq_model.encode_to_codebook(cond)
                
                self.cond_model = self.cond_model.cuda(0)
                text_cond = self.cond_model(text).cuda(self.global_rank)
                
                # use for saving training samples
                image_sample = image.detach()
                cond_sample = cond.detach()

                # TODO: generate noise
                rnd_gen = torch.Generator(device=f'cuda:{self.global_rank}').manual_seed(args.seed)
                noise = torch.randn(1, CA_configs['out_channels'],
                                    CA_configs['size'], CA_configs['size'],
                                    generator=rnd_gen, device=f'cuda:{self.global_rank}')

                # TODO: forward diffusion model to extract the internal representations
                t = torch.randint(0, self.diffusion_steps, (image.size(0),), device=f'cuda:{self.global_rank}').long()
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
                
                # record tensorboard logs
                self.logger.add_scalar('train/loss_per_step', loss.item(), self.iteration)

                if self.iteration % args.print_freq == 0:
                    print(f"Iteration: {self.iteration}/{args.max_steps}, Epoch: {epoch}/{args.epochs}, " "Loss: %.2f" % loss.item())
                    
                # TODO: save model checkpoints
                if self.iteration % args.checkpoint_freq == 0:
                    print("\nSaving checkpoints...\n")
                    os.makedirs(os.path.join(args.logdir, 'checkpoints'), exist_ok=True)
                    save_path = os.path.join(args.logdir, 'checkpoints', f"epoch_{epoch}_iters_{self.iteration}.pth")
                    torch.save({'model_state_dict': self.model.state_dict()}, save_path)
                    print(f"\nSuccessfully save checkpoint epoch_{epoch}_iters_{self.iteration}.pth!\n")
                    
                # TODO: save visualization
                if self.iteration % args.sample_freq == 0:
                    print("\nSaving training samples...\n")
                    cond = cond.detach()
                    cond_pred = cond_pred.detach()
                        
                    image_sample = self.vq_model.decode(image_sample)
                    cond_sample = self.vq_model.decode(cond_sample)
                    cond_pred = self.vq_model.decode(cond_pred)
                    
                    item_dict = {}
                    item_dict['image'] = image_sample
                    item_dict['cond'] = cond_sample
                    item_dict['cond_pred'] = cond_pred
                    
                    # visualize
                    self.visualize(item_dict=item_dict, epoch=epoch)
                    
                # TODO: evaluation
                if args.eval_freq > 0 and self.iteration % args.eval_freq == 0:
                    self.evaluate_t2i()
                
                self.logger.add_scalar('train/loss_per_epoch', loss.item(), epoch)
                
                # TODO: kill the process if training reaches maximum steps
                if self.iteration > args.max_steps:
                    save_path = os.path.join(args.logdir, 'checkpoints', f"LAST_epoch_{epoch}.pth")
                    torch.save({'model_state_dict': self.model.state_dict()}, save_path)
                    cleanup()
                    sys.exit(0)
            
            
    # TODO: train from unconditional diffusion model
    def U2I_training(self):
        args = self.args
        CA_configs = self.model_configs['condition_adaptor_config']
        
        train_dataset = U2ICollectedDataset(
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
                image, cond = batch['image'].cuda(self.global_rank), batch['cond'].cuda(self.global_rank)
                
                # TODO: foward vector quantatize model
                image = self.vq_model.encode_to_codebook(image)
                cond = self.vq_model.encode_to_codebook(cond)
                
                # use for saving training samples
                image_sample = image.detach()
                cond_sample = cond.detach()

                # TODO: generate noise
                rnd_gen = torch.Generator(device=f'cuda:{self.global_rank}').manual_seed(args.seed)
                noise = torch.randn(1, CA_configs['out_channels'], CA_configs['size'], CA_configs['size'], generator=rnd_gen, device=f'cuda:{self.global_rank}')

                # TODO: forward diffusion model to extract the internal representations
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
                
                # record tensorboard logs
                self.logger.add_scalar('train/loss_per_step', loss.item(), self.iteration)

                if self.iteration % args.print_freq == 0:
                    print(f"Iteration: {self.iteration}/{args.max_steps}, Epoch: {epoch}/{args.epochs}, " "Loss: %.2f" % loss.item())
                    
                # TODO: save model checkpoints
                if self.iteration % args.checkpoint_freq == 0:
                    print("\nSaving checkpoints...\n")
                    os.makedirs(os.path.join(args.logdir, 'checkpoints'), exist_ok=True)
                    save_path = os.path.join(args.logdir, 'checkpoints', f"epoch_{epoch}_iters_{self.iteration}.pth")
                    torch.save({'model_state_dict': self.model.state_dict()}, save_path)
                    print(f"\nSuccessfully save checkpoint epoch_{epoch}_iters_{self.iteration}.pth!\n")
                    
                # TODO: save visualization
                if self.iteration % args.sample_freq == 0:
                    print("\nSaving training samples...\n")
                    cond = cond.detach()
                    cond_pred = cond_pred.detach()
                        
                    image_sample = self.vq_model.decode(image_sample)
                    cond_sample = self.vq_model.decode(cond_sample)
                    cond_pred = self.vq_model.decode(cond_pred)
                    
                    item_dict = {}
                    item_dict['image'] = image_sample
                    item_dict['cond'] = cond_sample
                    item_dict['cond_pred'] = cond_pred
                    
                    # visualize
                    self.visualize(item_dict=item_dict, epoch=epoch)
                
                self.logger.add_scalar('train/loss_per_epoch', loss.item(), epoch)
                
                # TODO: kill the process if training reaches maximum steps
                if self.iteration > args.max_steps:
                    save_path = os.path.join(args.logdir, 'checkpoints', f"LAST_epoch_{epoch}.pth")
                    torch.save({'model_state_dict': self.model.state_dict()}, save_path)
                    cleanup()
                    sys.exit(0)


    # TODO: function of saving visualization
    def visualize(self, item_dict, epoch):
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
        
        cv2.imwrite(os.path.join(self.args.logdir, 'samples', f'epoch_{epoch}_iters_{self.iteration}.png'), item)

   
    # TODO: sample singe image with trained condition adaptor from TEXT-TO-IMAGE DIFFUSION MODELS
    def T2I_sampling_single_image_with_CA(self, args, input_path, cond_fn):
        CA_configs = self.model_configs['condition_adaptor_config']
        
        # TODO: initialize the models
        model = self.model
        diffusion_model = self.diffusion_model
        
        sampler = DDIMSampler(diffusion_model)
        vq_model = self.vq_model
        cond_stage_model = self.cond_model
        
        # TODO: initialize criterion for condition adaptor
        criterion = nn.MSELoss()
        
        # start evaluation
        with torch.no_grad():
            with diffusion_model.ema_scope():
                count = 0
                print("\nStart sampling...\n")
                    
                filename = os.path.basename(input_path).split('.')[0]
                
                # TODO - LOAD IN THE DATA
                
                # read in target condition
                target_cond = cv2.imread(input_path)    
                target_cond = cv2.resize(target_cond, (CA_configs['image_size'], CA_configs['image_size']), interpolation=cv2.INTER_NEAREST)
                
                if CA_configs['cond_type'] == 'stroke' or CA_configs['cond_type'] == 'image':
                    target_cond = cv2.cvtColor(target_cond, cv2.COLOR_RGB2BGR)
                if CA_configs['cond_type'] == 'edge':
                    _, target_cond = cv2.threshold(target_cond, 127.5, 255.0, cv2.THRESH_BINARY)
                if CA_configs['cond_type'] == 'saliency':
                    _, target_cond = cv2.threshold(target_cond, 127.5, 255.0, cv2.THRESH_BINARY)
                    
                target_cond = torch.from_numpy(target_cond.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).cuda(self.global_rank)
                target_cond = vq_model.encode_to_codebook(target_cond)                
                
                # TODO: forward text condition model - CLIP
                text_cond = cond_stage_model(self.args.caption)                
                    
                # TODO: classifier-free guidance
                if self.args.unconditional_guidance_scale != 1.0:
                    uc = diffusion_model.get_learned_conditioning(text_cond.shape[0] * [""])

                # shape of Gaussian noise
                shape = (self.args.channels, self.args.height // self.args.downsampled_factor, self.args.width // self.args.downsampled_factor)
                
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
                                        truncation_steps=self.args.truncation_steps,)
                
                # TODO: unpack returned outputs
                samples_ddim, _ = outputs
                    
                x_samples_ddim = diffusion_model.decode_first_stage(samples_ddim)

                import einops
                x_samples = (einops.rearrange(x_samples_ddim, 'b c h w -> b h w c') * 127.5 + 127.5).squeeze(0).cpu().numpy().clip(0, 255).astype(np.uint8)
                x_samples = cv2.cvtColor(x_samples, cv2.COLOR_BGR2RGB)
                
                cv2.imwrite(os.path.join(args.outdir, filename + '.png'), x_samples)


    # TODO: sample single image with trained condition adaptor from UNCONDITIONAL DIFFUSION MODELS
    def U2I_sampling_single_image_with_CA(self, args, input_path, cond_fn):
        CA_configs = self.model_configs['condition_adaptor_config']
        
        # TODO: initialize the models
        model = self.model
        diffusion_model = self.diffusion_model
        sampler = DDIMSampler(diffusion_model)
        vq_model = self.vq_model
        
        # TODO: initialize criterion for condition adaptor
        criterion = nn.MSELoss()
        
        # start evaluation
        with torch.no_grad():
            with diffusion_model.ema_scope():
                count = 0
                print("\nStart sampling...\n")
                    
                filename = os.path.basename(input_path).split('.')[0]
        
                # TODO: read in target condition from specific path
                target_cond = cv2.imread(input_path)    
                target_cond = cv2.resize(target_cond, (CA_configs['image_size'], CA_configs['image_size']))
                
                if CA_configs['cond_type'] == 'stroke' or 'image':
                    target_cond = cv2.cvtColor(target_cond, cv2.COLOR_BGR2RGB)
                if CA_configs['cond_type'] == 'edge':
                    _, target_cond = cv2.threshold(target_cond, 180, 255.0, cv2.THRESH_BINARY)
                if CA_configs['cond_type'] == 'saliency':
                    _, target_cond = cv2.threshold(target_cond, 127.5, 255.0, cv2.THRESH_BINARY)
                    
                target_cond = torch.from_numpy(target_cond.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).cuda(self.global_rank)
                
                target_cond_vis = target_cond.detach()
                
                target_cond = vq_model.encode_to_codebook(target_cond)                              

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
                                        truncation_steps=self.args.truncation_steps,)
                
                # TODO: unpack returned outputs
                samples_ddim, _ = outputs
                    
                x_samples_ddim = diffusion_model.decode_first_stage(samples_ddim)
                
                import einops
                x_samples = (einops.rearrange(x_samples_ddim, 'b c h w -> b h w c') * 127.5 + 127.5).squeeze(0).cpu().numpy().clip(0, 255).astype(np.uint8)
                x_samples = cv2.cvtColor(x_samples, cv2.COLOR_BGR2RGB)
                
                cv2.imwrite(os.path.join(args.outdir, filename + '.png'), x_samples)