import os
import torch

import clip
import torch.nn.functional as F
from torchmetrics import CLIPScore
from calculate_fid import calculate_fid_given_paths

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vit_version = "openai/clip-vit-large-patch14"
clip_score_criterion = CLIPScore(model_name_or_path=vit_version).to(device)

open_clip_version = "ViT-L/14"
# only using absolute path works
model_path = "/home/liuchang/.cache/clip/ViT-L-14.pt"
open_clip_model, open_clip_preprocess = clip.load(open_clip_version, device=device)

# TODO: Please make sure you have torchmetrics==0.11.4 installed
# calculate the correlation between generated samples and caption prompt
def calculate_clip_score(image, caption):
    """
    Calculate the CLIP score between generated samples and text prompt.
    The higher the score approaches 1.0, the more correlative that the samples and corresponding captions are.
    
    Input params:
        image: any image, in size of [B, C, H, W]
        caption: corresponding caption,
        
    Output params:
        score: CLIP score
        
    """
    score = clip_score_criterion(image, caption) / 100.0
    
    return score

# official implementation of T2I-Adapter
def calculate_open_clip_score(image, caption):
    """
    Calculate CLIP score between generated samples and text prompt, using the official implementation of T2I-Adapter.
    
    Input params:
        image: input image
        caption: corresponding caption
    
    Output params:
        score: CLIP score
    """
    image_features = open_clip_model.encode_image(image)
    text_features = open_clip_model.encode_text(caption)
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    score = F.cosine_similarity(image_features, text_features)
    score = score.cpu().data.numpy().astype(np.float32)
    
    return score

# calculate the statistics distance between images and samples, measuring the diversity in samples
def calculate_fid_score(image_path, sample_path):
    """
    Calculate the FID distance between generated samples and original images,
    measuring the diversity of generated samples.
    The lower the FID value is, the more diversified the generated samples are.
    
    Input params: 
        image_path: path string of original images,
        sample_path: path string of generated samples,
        
    Output params:
        score: FID score
    """
    paths = [image_path, sample_path]
    
    score = calculate_fid_given_paths(
        paths,
        batch_size=1,
        device=device,
        dims=2048,
        num_workers=8,
    )

    return score

# read paths as a list from a local path
def get_files(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(os.path.join(root, filespath))
    return ret

# dataset for calculating inception score
class InceptionScoreDataset(torch.utils.data.Dataset):
    def __init__(self, size, input_path):
        self.size = size
        self.input_path = input_path
        self.input_paths = self.get_files(input_path)

    def __getitem__(self, index):
        image = cv2.imread(self.input_paths[index])
        image = cv2.resize(image, (self.size, self.size))
        image = torch.from_numpy(image.astype(np.float32) / 127.5 - 1.0).permute(2, 1, 0)
        
        return image
    
    def get_files(self, path):
        # read a folder, return the complete path
        ret = []
        for root, dirs, files in os.walk(path):
            for filespath in files:
                ret.append(os.path.join(root, filespath))
        return ret

    def __len__(self):
        return len(self.input_paths)


if __name__ == "__main__":
    import torch
    import cv2
    import numpy as np
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
    
    # define configurations
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image_path", type=str, default='', help='path of original images')
    parser.add_argument("--caption_path", type=str, default='', help='path of corresponding captions of the source images')
    parser.add_argument("--sample_path", type=str, help='path of generated samples')
    parser.add_argument("--sd_version", default="1.4", type=str, help='stable diffusion version')
    parser.add_argument("--outdir", default=None, required=True, type=str, help='path to store the txt files')
    parser.add_argument('--size', default=512, type=int, help='size of source images and generated samples')
    parser.add_argument("--use_torchmetrics", type=bool, default=False, help='use implementations of torchmetrics to calculate CLIP score')
    parser.add_argument("--not_calculate_fid", action='store_true', help='calculate FID score')
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    experimental_setting = args.sample_path.split('/')[-2] if args.sample_path[-1] == "/" else args.sample_path.split('/')[-1]
   
    with torch.no_grad():
        
        # calculate FID score
        if not args.not_calculate_fid:
            print('\nStart calculating FID score...\n')
            fid_score = calculate_fid_score(args.image_path, args.sample_path)
        
        sample_paths = get_files(args.sample_path)
        
        print('\nFinish calculating FID score.\n')
        
        count = 0
        avg_clip_score = 0.0
        
        
        # calculate CLIP score
        print('\nStart calculating CLIP score...\n')
        
        for path in sample_paths:
            print(experimental_setting)
            count += 1
            print(f"Progress: {count}/{len(sample_paths)}")
            
            filename = os.path.basename(path).split('.')[0]
            
            # read in caption
            with open(os.path.join(args.caption_path, filename + '.txt')) as f:
                for line in f.readlines():
                    caption = line
                f.close()
            
            # read in image
            from PIL import Image
            sample = open_clip_preprocess(Image.open(path)).unsqueeze(0).to(device)
            caption = clip.tokenize(caption).to(device)
            
            # calculate CLIP score          
            # TODO: official version of T2I-Adapter
            clip_score = calculate_open_clip_score(sample, caption)
            
            torch.cuda.empty_cache()
            
            avg_clip_score += clip_score
        
        print('\nFinish calculating CLIP score.\n')
            
        # calculate average CLIP score
        avg_clip_score /= len(sample_paths)
        
        # write the calculated CLIP score
        txt = open(os.path.join(args.outdir, 'scores.txt'), "w")
        write_content = f"FID score: {fid_score}\nCLIP score: {avg_clip_score}" if not args.not_calculate_fid else f"Experimental Setting: {experimental_setting}.\nAverage CLIP score: {avg_clip_score}"
        txt.write(write_content)
        txt.close()
        
        print('Calculation done.')
        
        