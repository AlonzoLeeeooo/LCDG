from fileinput import filename
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


class LMDBSketchToImageGenerationTrain(Dataset):
    
    def __init__(self, size, image_path, sketch_path, is_binary=False, config=None):
        self.size = size
        self.is_binary = is_binary
        self.config = config or OmegaConf.create()
        self.img_env = lmdb.open(image_path, subdir=os.path.isdir(image_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        self.sketch_env = lmdb.open(sketch_path, subdir=os.path.isdir(sketch_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)


    def __len__(self):
        return 100000
    

    def __getitem__(self, i):
        
        # image lmdb reader
        with self.img_env.begin(write=False) as img_txn:
            img_byteflow = img_txn.get(str(i).encode())
        
        np_img = np.frombuffer(img_byteflow, np.uint8)
        image = np_img.reshape(256, 256, 3)
        
        # sketch lmdb reader
        with self.sketch_env.begin(write=False) as sketch_txn:
            sketch_byteflow = sketch_txn.get(str(i).encode())
        
        np_sketch = np.frombuffer(sketch_byteflow, np.uint8)
        sketch = np_sketch.reshape(256, 256, 3)
        
        # preprocess image
        image = cv2.resize(image, (self.size, self.size))
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image)
        
        # preprocess sketch
        sketch = cv2.resize(sketch, (self.size, self.size))
        
        if self.is_binary:
            _, sketch = cv2.threshold(sketch, 180.0, 255.0, cv2.THRESH_BINARY)
        
        sketch = sketch.astype(np.float32) / 255.0
        sketch = torch.from_numpy(sketch)
        sketch = torch.sum(sketch / 3, dim=2, keepdim=True)

        batch = {"image": image, "sketch": sketch}
        
        # normalize
        for k in batch:
            batch[k] = batch[k] * 2.0 - 1.0

        return batch


class SketchToImageGenerationTrain(Dataset):
    def __init__(self, size, image_path, sketch_path, is_binary=False, is_single_channel=False, config=None):
        self.size = size
        self.is_single_channel = is_single_channel
        self.is_binary = is_binary
        self.config = config or OmegaConf.create()
        self.sketch_path = sketch_path
        self.image_flist = self.get_files(image_path)
        

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
        filename = os.path.basename(self.image_flist[i]).split('.')[0] + '.png'
        sketch = np.array(Image.open(os.path.join(self.sketch_path, filename)).convert("RGB"))
        
        # preprocess image
        image = cv2.resize(image, (self.size, self.size))
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image)
        
        # preprocess sketch
        sketch = cv2.resize(sketch, (self.size, self.size))
        
        if self.is_binary:
            _, sketch = cv2.threshold(sketch, 180.0, 255.0, cv2.THRESH_BINARY)
            
        sketch = sketch.astype(np.float32) / 255.0
        sketch = torch.from_numpy(sketch)
        
        if self.is_single_channel:
            sketch = torch.sum(sketch / 3, dim=2, keepdim=True)

        batch = {"image": image, "sketch": sketch}
        
        # normalize
        for k in batch:
            batch[k] = batch[k] * 2.0 - 1.0

        return batch


class GrayscaleSketchToImageGenerationTrain(Dataset):
    def __init__(self, size, image_path, sketch_path, is_binary, config=None):
        self.size = size
        self.is_binary = is_binary
        self.config = config or OmegaConf.create()
        self.sketch_path = sketch_path
        self.image_flist = self.get_files(image_path)
        

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
        filename = os.path.basename(self.image_flist[i]).split('.')[0] + '.png'
        sketch = np.array(Image.open(os.path.join(self.sketch_path, filename)).convert("RGB"))
        
        # preprocess image
        image = cv2.resize(image, (self.size, self.size))
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image)
        
        # preprocess sketch
        sketch = cv2.resize(sketch, (self.size, self.size))
        
        if self.is_binary:
            _, sketch = cv2.threshold(sketch, 180.0, 255.0, cv2.THRESH_BINARY)
            
        sketch = sketch.astype(np.float32) / 255.0
        sketch = torch.from_numpy(sketch)

        batch = {"image": image, "sketch": sketch}
        
        # normalize
        for k in batch:
            batch[k] = batch[k] * 2.0 - 1.0

        return batch
    

if __name__ == '__main__':
    """
    import lmdb
    import cv2
    import os
    import numpy as np
    
    # test lmdb correspondence
    image_path = '/data/liuchang/Datasets/coco-stuff'
    img_env = lmdb.open(image_path, subdir=os.path.isdir(image_path),
                        readonly=True, lock=False,
                        readahead=False, meminit=False)
    with img_env.begin(write=False) as img_txn:
            img_byteflow = img_txn.get(str(2).encode())
    np_img = np.frombuffer(img_byteflow, np.uint8)  
    image = np_img.reshape(256, 256, 3)
    cv2.imwrite("img.png", image)
    
    image_path = '/data/liuchang/Datasets/coco-stuff-binary-sketch'
    img_env = lmdb.open(image_path, subdir=os.path.isdir(image_path),
                        readonly=True, lock=False,
                        readahead=False, meminit=False)
    with img_env.begin(write=False) as img_txn:
        img_byteflow = img_txn.get(str(2).encode())
    np_img = np.frombuffer(img_byteflow, np.uint8)  
    image = np_img.reshape(256, 256, 3)
    _, image = cv2.threshold(image, 180.0, 255.0, cv2.THRESH_BINARY)
    cv2.imwrite("sketch.png", image)
    print("Done.")
    """
    import lmdb
    import numpy as np
    import torch
    from torch.utils.data import Dataset, DataLoader
    
    image_path = '/data/liuchang/Datasets/celebahq'
    sketch_path = '/data/liuchang/Datasets/celebahq-sketch'
    dataset = LMDBSketchToImageGenerationTrain(size=256, image_path=image_path, sketch_path=sketch_path, is_binary=False)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=8, shuffle=False)
    
    for i, batch in enumerate(dataloader):
        
        for key in batch.keys():
            batch[key] = (batch[key] + 1) / 2 * 255.0
        
        sketch = batch['sketch']
        sketch = torch.cat([sketch, sketch, sketch], dim=3)
        image = batch['image'].cpu().numpy().astype(np.uint8).squeeze(0)
        sketch = sketch.cpu().numpy().astype(np.uint8).squeeze(0)
        
        cv2.imwrite('img.png', image)
        cv2.imwrite('sketch.png', sketch)
        print('Done.')