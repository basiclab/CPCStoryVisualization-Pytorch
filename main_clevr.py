from __future__ import print_function
import torch.backends.cudnn as cudnn
import torch
import torchvision.transforms as transforms
import PIL
import argparse
import os
import random
import sys
import pdb
import pprint
import datetime
import dateutil
import dateutil.tz
import numpy as np
import functools
import datasets.clevr as data
import datasets.utils as video_transform
from tqdm import tqdm
dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

from miscc.datasets import TextDataset
from miscc.config import cfg, cfg_from_file
from miscc.utils import mkdir_p
from trainer import GANTrainer
from inference import Infer

def video_transform(video, image_transform):
    vid = []
    for im in video:
        vid.append(image_transform(im))

    vid = torch.stack(vid).permute(1, 0, 2, 3)

    return vid


def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN network')
    parser.add_argument('--debug', default=False)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/clevr.yml', type=str)
    parser.add_argument('--load_ckpt', default=None, type=str)
    parser.add_argument('--continue_ckpt', default=None, type=str)
    parser.add_argument('--gpu',  dest='gpu_id', type=str, default='')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--eval_fid', type=bool, default=False)
    parser.add_argument('--eval_fvd', type=bool, default=False)
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    print('Using config:')
    pprint.pprint(cfg)
    random.seed(0)
    torch.manual_seed(0)
    dir_path = cfg.DATA_DIR
    if cfg.CUDA:
        torch.cuda.manual_seed_all(0)
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    if args.debug:
        output_dir = './output/debug'
    else:
        output_dir = './output/{}'.format(cfg.CONFIG_NAME)
        
    num_gpu = len(cfg.GPU_ID.split(','))
    use_segment = cfg.SEGMENT_LEARNING

    if cfg.CASCADE_MODEL:
        cfg.SEGMENT_LEARNING = True

    if cfg.TRAIN.FLAG:
        image_transforms = transforms.Compose([
            PIL.Image.fromarray,
            transforms.Resize((cfg.IMSIZE, cfg.IMSIZE) ),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        image_transforms_seg = transforms.Compose([
            PIL.Image.fromarray,
            transforms.Resize((cfg.SESIZE, cfg.SESIZE) ),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])])
            #transforms.Normalize(mean=0.5, std=0.5)])

        video_len = 5
        n_channels = 3
        video_transforms = functools.partial(video_transform, image_transform=image_transforms) # Only need to feed video later

        # Train dataset
        storydataset = data.StoryDataset(dir_path, video_transforms, is_train=True)
        imagedataset = data.ImageDataset(dir_path, image_transforms, image_transforms_seg, 
            use_segment=use_segment, segment_name=cfg.TRAIN.SEGMENT_NAME)

        imageloader = torch.utils.data.DataLoader(
            imagedataset, batch_size=cfg.TRAIN.IM_BATCH_SIZE * num_gpu,
            drop_last=True, shuffle=True, num_workers=int(cfg.WORKERS))

        storyloader = torch.utils.data.DataLoader(
            storydataset, batch_size=cfg.TRAIN.ST_BATCH_SIZE * num_gpu,
            drop_last=True, shuffle=True, num_workers=int(cfg.WORKERS))


        # Test dataset
        testdataset = data.StoryDataset(dir_path, video_transforms, is_train=False)

        testloader = torch.utils.data.DataLoader(
            testdataset, batch_size=cfg.TRAIN.ST_BATCH_SIZE * num_gpu,
            drop_last=True, shuffle=False, num_workers=int(cfg.WORKERS))

        if args.eval_fid:
            algo = Infer(output_dir, 1.0)
            algo.eval_fid2(testloader, video_transforms, image_transforms)

        elif args.eval_fvd:
            algo = Infer(output_dir, 1.0)
            algo.eval_fvd(imageloader, storyloader, testloader, cfg.STAGE)

        elif args.load_ckpt != None:
            # For inference training result
            algo = Infer(output_dir, 1.0, args.load_ckpt)
            algo.inference(imageloader, storyloader, testloader, cfg.STAGE)
        else:
            # For training model
            algo = GANTrainer(output_dir, args, ratio=1.0)
            algo.train(imageloader, storyloader, testloader, cfg.STAGE)
    else:
        datapath= '%s/test/val_captions.t7' % (cfg.DATA_DIR)
        algo = GANTrainer(output_dir)
        algo.sample(datapath, cfg.STAGE)

