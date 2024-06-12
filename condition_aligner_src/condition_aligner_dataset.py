import cv2
import sys
import os
import torch
import numpy as np
from torch.utils.data import Dataset

    
class ImageTextConditionDataset(Dataset):
    def __init__(self, cond_type, image_dir, cond_dir, text_dir, image_size, kmeans_center=None):
        super().__init__()
        self.image_size = image_size
        self.cond_type = cond_type                                   # edge, saliency or strokes, to distinguish how to process the data
        self.cond_dir = cond_dir
        self.text_dir = text_dir
        self.image_paths = self.get_files_from_txt(image_dir)
        self.image_paths = sorted(self.image_paths)
        

        # kmeans for stroke generation
        # k-means for stroke generation
        if self.cond_type == 'stroke':
            assert kmeans_center is not None
            self.kmeans_center = kmeans_center
            self.criteria = (cv2.TERM_CRITERIA_EPS + 
                             cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            self.flags = cv2.KMEANS_RANDOM_CENTERS

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
                
        # load in image
        image = cv2.imread(self.image_paths[index])
        filename = os.path.basename(self.image_paths[index]).split('.')[0]
        
        if self.cond_type == 'stroke':
            filtered_image = cv2.medianBlur(image, ksize=23)

            # k-means
            compactness, label, center = cv2.kmeans(np.float32(filtered_image.reshape(-1, 3)),
                                                    self.kmeans_center,
                                                    None, self.criteria, 10, self.flags)

            center = np.uint8(center)
            cond = center[label.flatten()]
            cond = cond.reshape((image.shape))
        
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.cond_type == "canny":
            cond = cv2.Canny(image, 200, 225.0)
        
        image = torch.from_numpy(image.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)                  # [H, W, C] -> [C, H, W]
        
        # load correpsonding condition
        if self.cond_type == 'image' or self.cond_type == "style":
            cond = cv2.imread(self.image_paths[index])
        elif self.cond_type == 'edge' or self.cond_type == "saliency":
            cond = cv2.imread(os.path.join(self.cond_dir, filename + '.png'))                          # saliency, edge

        cond = cv2.resize(cond, (self.image_size, self.image_size))
        
        # only binarize condition of ``edge'' or ``saliency''
        # switch BGR channels of color strokes into RGB channels
        if self.cond_type == "edge":
            _, cond = cv2.threshold(cond, thresh=180.0, maxval=255.0, type=cv2.THRESH_BINARY)              # to binarize
        elif self.cond_type == "saliency":
            _, cond = cv2.threshold(cond, thresh=127.5, maxval=255.0, type=cv2.THRESH_BINARY)              # to binarize
        elif self.cond_type == "stroke" or self.cond_type == 'image' or self.cond_type == "style":
            cond = cv2.cvtColor(cond, cv2.COLOR_BGR2RGB)    

        if self.cond_type == "style":
            pixel_cond = torch.from_numpy(cond.astype(np.float32) / 255.0).permute(2, 0, 1).detach()
        
        if self.cond_type == "canny":
            cond = torch.from_numpy(cond.astype(np.float32) / 127.5 - 1.0).unsqueeze(2).permute(2, 0, 1)       # [H, W, C] -> [C, H, W]
            cond = torch.cat([cond, cond, cond], dim=0)
        else:
            cond = torch.from_numpy(cond.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)                    # [H, W, C] -> [C, H, W]
        
        with open(os.path.join(self.text_dir, filename + '.txt')) as f:
            for line in f.readlines():
                text = line
            # for verbose only
            if f.readlines() == []:
                text = ""
        f.close()
        
        # warp in batch dict
        batch = {}
        batch['image'] = image
        batch['cond'] = cond
        batch['text'] = text
        
        if self.cond_type == "style":
            batch['pixel_cond'] = pixel_cond
        
        return batch

    def get_files_from_txt(self, path):
        file_list = []
        f = open(path)
        for line in f.readlines():
            line = line.strip("\n")
            file_list.append(line)
            sys.stdout.flush()
        f.close()

        return file_list

    def get_files_from_path(self, path):
        ret = []
        for root, dirs, files in os.walk(path):
            for filespath in files:
                ret.append(os.path.join(root, filespath))
        return ret
    
    
class ImageConditionDataset(Dataset):
    def __init__(self, cond_type, image_dir, cond_dir, image_size):
        super().__init__()
        self.image_size = image_size
        self.cond_type = cond_type                                   # edge, saliency or strokes, to distinguish how to process the data
        self.cond_dir = cond_dir
        self.image_paths = self.get_files_from_txt(image_dir)
        self.image_paths = sorted(self.image_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        
        # load in image
        image = cv2.imread(self.image_paths[index])
        filename = os.path.basename(self.image_paths[index]).split('.')[0]
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)                  # [H, W, C] -> [C, H, W]

        # load correpsonding condition
        if self.cond_type == "image" or "style":
            cond = cv2.imread(self.image_paths[index])
        else:
            cond = cv2.imread(os.path.join(self.cond_dir, filename + '.png'))
        cond = cv2.resize(cond, (self.image_size, self.image_size))
        
        # preprocess conditions - binarize or transfer to RGB channels
        if self.cond_type == "image" or self.cond_type == "stroke" or self.cond_type == "style":
            cond = cv2.cvtColor(cond, cv2.COLOR_BGR2RGB)
        elif self.cond_type == "edge":
            _, cond = cv2.threshold(cond, thresh=180.0, maxval=255.0, type=cv2.THRESH_BINARY)              # to binarize
        elif self.cond_type == "saliency":
            _, cond = cv2.threshold(cond, thresh=127.5, maxval=255.0, type=cv2.THRESH_BINARY)              # to binarize
            
        cond = torch.from_numpy(cond.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)                    # [H, W, C] -> [C, H, W]

        # warp in batch dict
        batch = {}
        batch['image'] = image
        batch['cond'] = cond
        
        return batch

    def get_files_from_txt(self, path):
        file_list = []
        f = open(path)
        for line in f.readlines():
            line = line.strip("\n")
            file_list.append(line)
            sys.stdout.flush()
        f.close()

        return file_list

    def get_files_from_path(self, path):
        ret = []
        for root, dirs, files in os.walk(path):
            for filespath in files:
                ret.append(os.path.join(root, filespath))
        return ret
    