from fileinput import filename
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional as F
import time
import os
import sys
import cv2
import bdcn
import argparse
import os

def get_files(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(os.path.join(root, filespath))
    return ret

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

def make_dir(data_dir):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)


def test(model, args):
    test_root = args.indir
    test_lst = sorted(get_files(test_root))
    save_sideouts = 0
    if save_sideouts:
        for j in xrange(5):
            make_dir(os.path.join(save_dir, 's2d_'+str(k)))
            make_dir(os.path.join(save_dir, 'd2s_'+str(k)))
    mean_bgr = np.array([104.00699, 116.66877, 122.67892])
    save_dir = args.outdir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if args.cuda:
        model.cuda()
    model.eval()
    start_time = time.time()
    all_t = 0
    count = 0
    for nm in test_lst:
        sys.stdout.flush()
        print(f"{count+1}/{len(test_lst)}")
        # nm = osp.basename(nm).split('.')[0]
        res_dir = args.outdir

        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        
        filename = os.path.basename(test_lst[count])
        filename = filename.split('.')[0] + '.png'

        data = cv2.imread(test_lst[count])
        count += 1
        data = cv2.resize(data, [args.size, args.size])
        # if data is None: continue
        # print(os.path.join(test_root, nm))
        # data = cv2.resize(data, (data.shape[1]/2, data.shape[0]/2), interpolation=cv2.INTER_LINEAR)
        data = np.array(data, np.float32)
        data = data - mean_bgr
        data = data.transpose((2, 0, 1))
        data = torch.from_numpy(data).float().unsqueeze(0)
        if args.cuda:
            data = data.cuda()
        data = Variable(data)
        t1 = time.time()
        out = model(data)
        if '/' in nm:
            nm = nm.split('/')[-1]
        if save_sideouts:
            out = [F.sigmoid(x).cpu().data.numpy()[0, 0, :, :] for x in out]
            k = 1
            for j in xrange(5):
                # savemat(osp.join(save_dir, 's2d_'+str(k), nm+'.mat'), {'prob': out[j]})
                cv2.imwrite(os.path.join(save_dir, 's2d_'+str(k), '%s.jpg'%nm[i]), 255-t*255)
                # savemat(osp.join(save_dir, 'd2s_'+str(k), nm+'.mat'), {'prob': out[j+5]})
                cv2.imwrite(os.path.join(save_dir, 'd2s_'+str(k), '%s.jpg'%nm), 255-255*t)
                k += 1
        else:
            out = [F.sigmoid(out[-1]).cpu().data.numpy()[0, 0, :, :]]
        cv2.imwrite(os.path.join(res_dir, filename), 255*out[-1])
        all_t += time.time() - t1
    print(all_t)
    print('Overall Time use: ', time.time() - start_time)

def main():
    import time
    print(time.localtime())
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    model = bdcn.BDCN()
    model.load_state_dict(torch.load('%s' % (args.model)))
    # print model.fuse.weight.data, model.fuse.bias.data
    print(model.fuse.weight.data)
    test(model, args)

def parse_args():
    parser = argparse.ArgumentParser('test BDCN')
    parser.add_argument('-c', '--cuda', action='store_true',
        help='whether use gpu to train network')
    parser.add_argument('-g', '--gpu', type=str, default='0',
        help='the gpu id to train net')
    parser.add_argument('-m', '--model', type=str, default='checkpoints/bdcn.pth',
        help='the model to test')
    parser.add_argument('--outdir', type=str, default='result',
        help='the dir to store result')
    parser.add_argument('--indir', type=str, default='./')
    parser.add_argument('--data_prefix', type=str, default='')
    parser.add_argument('--size', type=int, default=256)
    return parser.parse_args()

if __name__ == '__main__':
    main()
