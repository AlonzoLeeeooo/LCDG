import os
import cv2
import argparse
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
    parser.add_argument('--outdir', default='generated_platte', type=str)
    parser.add_argument('--threshold1', default=200, type=int)
    parser.add_argument('--threshold2', default=225, type=int)

    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    input_flist = get_files(args.indir)

    count = 0
    for path in input_flist:
        count += 1
        
        filename = os.path.basename(path)
    
        image = cv2.imread(path)
        canny_edge = cv2.Canny(image, args.threshold1, args.threshold2)
        output_path = os.path.join(args.outdir, filename.split('.')[0] + f'.png')
    
        cv2.imwrite(output_path, canny_edge)
        print(f'{count}/{len(input_flist)}')
