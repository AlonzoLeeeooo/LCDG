import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
import sys
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB

import cv2
import decord
from einops import rearrange

# funciton to return a path list from a directory
def get_files(path):
    file_list = []
    for root, dirs, files in os.walk(path):
        for filepath in files:
            file_list.append(os.path.join(root, filepath))

    return file_list

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

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(filename, pred, outdir, height, width):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np * 255).convert('RGB')
    imo = im.resize((height, width),resample=Image.BILINEAR)

    imo.save(os.path.join(outdir, filename + '.png'))

def main():

    # 1. Define basic configurations
    model_name = 'u2net'

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', default='', type=str, help='Input path of video frames')
    parser.add_argument('--outdir', default='', type=str, help='Output path of extracted mask of first frames')
    parser.add_argument('--model_dir', default='', type=str, help='Pre-trained model weights of U2-Net')

    args = parser.parse_args()

    image_dir = args.indir
    prediction_dir = args.outdir

    if not prediction_dir.endswith('/'):
        prediction_dir = prediction_dir + '/'
    model_dir = args.model_dir


    # 2. Define model
    if(model_name=='u2net'):
        net = U2NET(3,1)
    elif(model_name=='u2netp'):
        net = U2NETP(3,1)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()
    
    
    # 3. Inference
    count = 0
    for image_path in os.listdir(args.indir):
        # 3.1 Load in image
        image = cv2.imread(os.path.join(args.indir, image_path))
        filename = image_path.replace('.png', '').replace('.jpg', '')
        height, width, _ = image.shape
        
        # 3.2 Pass the first frame forward the model
        # The same as the original implementation of U2-Net
        inputs_test = image
        inputs_test = torch.from_numpy(inputs_test.astype(np.float32) / 255.).unsqueeze(0).permute(0, 3, 1, 2)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)
        pred = d1[:,0,:,:]
        pred = normPRED(pred)

        # 3.3 Save outputted mask
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        save_output(filename, pred, prediction_dir, height=height, width=width)

        del d1,d2,d3,d4,d5,d6,d7

if __name__ == "__main__":
    main()