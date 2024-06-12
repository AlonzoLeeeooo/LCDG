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


class LMDBInpaintingTrain(Dataset):
    def __init__(self, size, data_root, config=None):
        self.size = size
        self.config = config or OmegaConf.create()
        self.env = lmdb.open(data_root, subdir=os.path.isdir(data_root),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)


    def generate_stroke_mask(self, im_size, parts=4, maxVertex=25, maxLength=80, maxBrushWidth=40, maxAngle=360):
        
        mask = np.zeros((im_size[0], im_size[1], 1), dtype=np.float32)
        for i in range(parts):
            mask = mask + self.np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, im_size[0], im_size[1])
        mask = np.minimum(mask, 1.0)

        return mask


    def np_free_form_mask(self, maxVertex, maxLength, maxBrushWidth, maxAngle, h, w):

        mask = np.zeros((h, w, 1), np.float32)
        numVertex = np.random.randint(maxVertex + 1)
        startY = np.random.randint(h)
        startX = np.random.randint(w)
        brushWidth = 0
        for i in range(numVertex):
            angle = np.random.randint(maxAngle + 1)
            angle = angle / 360.0 * 2 * np.pi
            if i % 2 == 0:
                angle = 2 * np.pi - angle
            length = np.random.randint(maxLength + 1)
            brushWidth = np.random.randint(10, maxBrushWidth + 1) // 2 * 2
            nextY = startY + length * np.cos(angle)
            nextX = startX + length * np.sin(angle)
            nextY = np.maximum(np.minimum(nextY, h - 1), 0).astype(np.int)
            nextX = np.maximum(np.minimum(nextX, w - 1), 0).astype(np.int)
            cv2.line(mask, (startY, startX), (nextY, nextX), 1, brushWidth)
            cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
            startY, startX = nextY, nextX
        cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
        
        return mask


    def get_files_from_txt(self, path):

        file_list = []
        f = open(path)
        for line in f.readlines():
            line = line.strip("\n")
            file_list.append(line)
            sys.stdout.flush()
        f.close()

        return file_list


    def get_files(self, path):

        # read a folder, return the complete path
        ret = []
        for root, dirs, files in os.walk(path):
            for filespath in files:
                ret.append(os.path.join(root, filespath))

        return ret


    def __len__(self):
        return 64000


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

        mask = np.array(Image.open("ldm/data/1.png").convert("L"))
        mask = cv2.resize(mask, (self.size, self.size))
        mask = mask.astype(np.float32) / 255.0
        mask = mask[None]
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask).permute(1, 2, 0)

        masked_image = (1 - mask) * image

        batch = {"image": image, "mask": mask, "masked_image": masked_image}
        for k in batch:
            batch[k] = batch[k] * 2.0 - 1.0

        return batch


class InpaintingTrain(Dataset):
    def __init__(self, size, data_root, config=None):
        self.size = size
        self.config = config or OmegaConf.create()
        self.image_flist = self.get_files_from_txt(data_root)


    def generate_stroke_mask(self, im_size, parts=4, maxVertex=25, maxLength=80, maxBrushWidth=40, maxAngle=360):
        
        mask = np.zeros((im_size[0], im_size[1], 1), dtype=np.float32)
        for i in range(parts):
            mask = mask + self.np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, im_size[0], im_size[1])
        mask = np.minimum(mask, 1.0)

        return mask


    def np_free_form_mask(self, maxVertex, maxLength, maxBrushWidth, maxAngle, h, w):

        mask = np.zeros((h, w, 1), np.float32)
        numVertex = np.random.randint(maxVertex + 1)
        startY = np.random.randint(h)
        startX = np.random.randint(w)
        brushWidth = 0
        for i in range(numVertex):
            angle = np.random.randint(maxAngle + 1)
            angle = angle / 360.0 * 2 * np.pi
            if i % 2 == 0:
                angle = 2 * np.pi - angle
            length = np.random.randint(maxLength + 1)
            brushWidth = np.random.randint(10, maxBrushWidth + 1) // 2 * 2
            nextY = startY + length * np.cos(angle)
            nextX = startX + length * np.sin(angle)
            nextY = np.maximum(np.minimum(nextY, h - 1), 0).astype(np.int)
            nextX = np.maximum(np.minimum(nextX, w - 1), 0).astype(np.int)
            cv2.line(mask, (startY, startX), (nextY, nextX), 1, brushWidth)
            cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
            startY, startX = nextY, nextX
        cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
        
        return mask


    def get_files_from_txt(self, path):

        file_list = []
        f = open(path)
        for line in f.readlines():
            line = line.strip("\n")
            file_list.append(line)
            sys.stdout.flush()
        f.close()

        return file_list


    def get_files(self, path):

        # read a folder, return the complete path
        ret = []
        for root, dirs, files in os.walk(path):
            for filespath in files:
                ret.append(os.path.join(root, filespath))

        return ret


    def __len__(self):
        return len(self.image_flist)


    def __getitem__(self, i):
        
        image = np.array(Image.open(self.image_flist[i]).convert("RGB"))
        image = cv2.resize(image, (self.size, self.size))
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image)

        mask = self.generate_stroke_mask([self.size, self.size])
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask)

        masked_image = (1 - mask) * image

        batch = {"image": image, "mask": mask, "masked_image": masked_image}
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