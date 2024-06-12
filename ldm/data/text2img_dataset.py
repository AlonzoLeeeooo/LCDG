import os
import numpy as np
import PIL
import cv2
import torch
import sys
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import random

class Text2ImageTrain(Dataset):
    def __init__(self,
                 size,
                 image_path,
                 caption_path,
                 ):

        self.caption_path = caption_path
        self.image_paths = self.get_files_from_txt(image_path)
        self.size = size
        

    def get_files_from_path(self, path):
    
        # read a folder, return the complete path
        ret = []
        for root, dirs, files in os.walk(path):
            for filespath in files:
                ret.append(os.path.join(root, filespath))

        return ret
    
    
    def get_files_from_txt(self, path):
    
        file_list = []
        f = open(path)
        for line in f.readlines():
            line = line.strip("\n")
            file_list.append(line)
            sys.stdout.flush()
        f.close()

        return file_list


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        batch = {}
        image = np.array(Image.open(self.image_paths[i]).convert('RGB'))
        filename = os.path.basename(self.image_paths[i]).split('.')[0]
        sub_folder = self.image_paths[i].split('/')[-2]
        
        # read in text
        with open(os.path.join(self.caption_path, sub_folder, filename + '.txt')) as f:
            for line in f.readlines():
                text = line
        f.close()
        
        batch['caption'] = text
        
        # read in image
        image = cv2.resize(image, (self.size, self.size))
        image = image.astype(np.float32) / 127.5 - 1.0
        image = torch.from_numpy(image)

        batch['image'] = image

        return batch