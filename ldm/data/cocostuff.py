import os, sys, yaml, pickle, shutil, tarfile, glob
import cv2
import albumentations
import PIL
import numpy as np
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from functools import partial
from PIL import Image
from tqdm import tqdm
import lmdb

import torch
from torch.utils.data import Dataset, Subset

class LMDBCOCOStuff(Dataset):
    def __init__(self, size, data_root, config=None):
        self.size = size
        self.config = config or OmegaConf.create()
        self.env = lmdb.open(data_root, subdir=os.path.isdir(data_root),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)

    def __len__(self):
        return 100000

    def __getitem__(self, i):

        # lmdb reader
        with self.env.begin(write=False) as txn:
            byteflow = txn.get(str(i).encode())
        
        np_img = np.frombuffer(byteflow, np.uint8)
        image = np_img.reshape(256, 256, 3)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.size, self.size))
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image)

        batch = {"image": image}
        for k in batch:
            batch[k] = batch[k] * 2.0 - 1.0

        return batch

if __name__ == '__main__':
    
    # test the free form mask algorithm
    func = InpaintingTrain(256, '/data1/liss/dataset/Places2Standard/flists/train_flist.txt')
    mask = func.generate_stroke_mask([256, 256])
    mask = np.concatenate([mask, mask, mask], axis=2)
    mask = (mask*255).astype(np.uint8)
    cv2.imwrite('1.png', mask)