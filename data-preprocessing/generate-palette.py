import os
import cv2
import argparse
import sys
from tqdm import tqdm

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', default='', type=str)
    parser.add_argument('--size', default=8, type=int)
    parser.add_argument('--image_resolution', default=512, type=int)
    parser.add_argument('--outdir', default='generated_platte', type=str)
    parser.add_argument('--visualize_intermediate', action='store_true')

    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    if args.visualize_intermediate:
        os.makedirs(os.path.join(args.outdir, 'intermediates'), exist_ok=True)
    input_list = get_files(args.indir)
    pbar = tqdm(total=len(input_list))
    
    for path in input_list:
        pbar.update(1)
        filename = os.path.basename(path)
        
        image = cv2.imread(path)
        if args.visualize_intermediate:
            cv2.imwrite(os.path.join(args.outdir, 'intermediates', filename.split('.')[0] + '-source.png'), image)
        
        h, w, _ = image.shape
        image = cv2.resize(image, (args.size, args.size), interpolation=cv2.INTER_CUBIC)
        
        if args.visualize_intermediate:
            cv2.imwrite(os.path.join(args.outdir, 'intermediates', filename.split('.')[0] + '-cubic.png'), image)
        
        image = cv2.resize(image, (args.image_resolution, args.image_resolution), interpolation=cv2.INTER_NEAREST)
        if args.visualize_intermediate:
            cv2.imwrite(os.path.join(args.outdir, 'intermediates', filename.split('.')[0] + '-nearest.png'), image)
        
        output_path = os.path.join(args.outdir, filename.split('.')[0] + '.png')
        
        cv2.imwrite(output_path, image)