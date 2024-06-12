import os
import cv2
import argparse
import numpy as np
import torch
import sys
import argparse


def get_files(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(os.path.join(root, filespath))
    return ret



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', default='', type=str)
    parser.add_argument('--outdir', default='', type=str)

    args = parser.parse_args()

    flist = get_files(args.indir)
    file_names = []
    count = 0
    for item in flist[]:
        count += 1
        sys.stdout.flush()
        print(f"Progress: [{count}/{len(flist)}]")
        file_names.append(item)

    os.makedirs(args.outdir, exist_ok=True)
    data_flist = open(os.path.join(args.outdir, 'data_flist.txt'), "w")

    for i in file_names:
        data_flist.write(i+'\n')

    data_flist.close()

    print(f"Total line number: {len(file_names)}.")
