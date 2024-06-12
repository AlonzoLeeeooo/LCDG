import argparse
import cv2
import os
import sys
import numpy as np


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
    parser.add_argument('--kmeans_center', default=6, type=int)
    parser.add_argument('--num', default=1000000000, type=int)
    parser.add_argument('--visualize_intermediate', action='store_true')

    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    os.makedirs(os.path.join(args.outdir, 'intermediates'), exist_ok=True)

    image_flist = get_files(args.indir)

    # configurations of k-means
    criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS

    count = 0
    for item in image_flist:
        count += 1
        if count > args.num:
            print("Stop generating...")
            sys.exit(0)
        
        print(f"Progress: {count}/{len(image_flist)}")

        filename = item.split('/')[-1]
        image = cv2.imread(item)
        
        if args.visualize_intermediate:
            cv2.imwrite(os.path.join(args.outdir, 'intermediates', filename.split('.')[0] + '-source.png'), image)
        
        image = cv2.resize(image, (256, 256))
        image = cv2.medianBlur(image, ksize=23)

        if args.visualize_intermediate:
            cv2.imwrite(os.path.join(args.outdir, 'intermediates',  filename.split('.')[0] + '-blur.png'), image)

        # k-means
        compactness, label, center = cv2.kmeans(np.float32(image.reshape(-1, 3)),
                                                args.kmeans_center,
                                                None, criteria, 10, flags)

        center = np.uint8(center)
        result = center[label.flatten()]
        result = result.reshape((image.shape)).astype(np.uint8)

        # save
        cv2.imwrite(os.path.join(args.outdir, filename.split('.')[0] + '.png'), result)