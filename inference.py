from __future__ import print_function
from six.moves import range
from PIL import Image

import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import time
import pdb
import numpy as np
import torchfile
import shutil
from tqdm import tqdm
from tqdm import TqdmSynchronisationWarning
from miscc.config import cfg
from miscc.utils import mkdir_p
from miscc.utils import weights_init
from shutil import copyfile
import glob
from torchvision import transforms
from tensorboardX import SummaryWriter
from miscc.datasets import FolderStoryDataset, FolderImageDataset
from fid.vfid_score import fid_score as vfid_score
from fid.fid_score_v import fid_score
from fid.utils import StoryGANDataset, IgnoreLabelDataset
from miscc.utils import inference_samples

class Infer(object):
    def __init__(self, output_dir, ratio, load_ckpt=None, save_img=True):
        self.load_ckpt = load_ckpt
        self.output_dir = output_dir
        self.log_dir = os.path.join(output_dir, 'log')
        self.model_dir = os.path.join(output_dir, 'Model')
        self.save_dir = "./Evaluation/{}".format(cfg.CONFIG_NAME)

        self.video_len = cfg.VIDEO_LEN
        s_gpus = cfg.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)
        self.imbatch_size = cfg.TRAIN.IM_BATCH_SIZE * self.num_gpus
        self.stbatch_size = cfg.TRAIN.ST_BATCH_SIZE * self.num_gpus
        self.ratio = ratio
        torch.cuda.set_device(self.gpus[0])
        cudnn.benchmark = True
        
        if not load_ckpt:
            # If not load ckpt, then do evaluation
            self._logger = SummaryWriter(self.log_dir)
        
        if cfg.TRAIN.FLAG and save_img:
            mkdir_p(self.save_dir)
    # ############# For training stageI GAN #############
    def load_network_stageI(self, output_dir, load_ckpt=None):
        import hashlib
        import importlib

        hash_code = str(int(hashlib.sha1(output_dir.encode('utf-8')).hexdigest(), 16) % (10 ** 8))
        if os.path.exists(os.path.join(output_dir, 'model.py')):
            copyfile(os.path.join(output_dir, 'model.py'), 'model_saved_{}.py'.format(hash_code))
            #import sys
            #sys.path.insert(1, os.path.join(output_dir))
            full_module_name = 'model_saved_{}'.format(hash_code)
            saved_model = importlib.import_module(full_module_name)
            StoryGAN = saved_model.StoryGAN
            # from model_saved import StoryGAN
        else:
            from model import StoryGAN
        
        netG = StoryGAN(self.video_len)
        netG.apply(weights_init)
        #print(netG)

        if load_ckpt != None:
            path_G = os.path.join(self.model_dir, "netG_epoch_{}.pth".format(load_ckpt))
            state_dict = torch.load(path_G, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load from: ', path_G)

        if cfg.CUDA:
            netG.cuda()
        return netG

    def calculate_vfid(self, netG, epoch, testloader):
        netG.eval()
        with torch.no_grad():
            eval_modeldataset = StoryGANDataset(netG, len(testloader), testloader.dataset)
            vfid_value = vfid_score(IgnoreLabelDataset(testloader.dataset),
                eval_modeldataset, cuda=True, normalize=True, r_cache=None
            )
            fid_value = fid_score(IgnoreLabelDataset(testloader.dataset),
                    eval_modeldataset, cuda=True, normalize=True, r_cache=None
                )
        #netG.train()
        if self._logger:
            self._logger.add_scalar('Off_Evaluation/vfid',  vfid_value,  epoch)
            self._logger.add_scalar('Off_Evaluation/fid',  fid_value,  epoch)

        return fid_value, vfid_value

    def calculate_fvd(self, gen_path, epoch, num_of_video):
        from fvd.fvd import calculate_fvd_from_inference_result

        fvd_value = calculate_fvd_from_inference_result(gen_path, num_of_video=num_of_video)
        print('[{}] {}----------'.format(epoch, fvd_value))
        if self._logger:
            self._logger.add_scalar('Off_Evaluation/fvd',  fvd_value,  epoch)
        return fvd_value


    def eval_fid(self, test_loader):
        output_score_filename = os.path.join(self.save_dir, 'fid_score.csv')
        models = os.listdir(self.model_dir)
        with open(output_score_filename, 'a') as f:
            f.write('epoch,fid,vfid\n')
        for epoch in range(121):
            if 'netG_epoch_{}.pth'.format(epoch) in models:
                print('Evaluating epoch {}'.format(epoch))
                netG = self.load_network_stageI(self.output_dir, load_ckpt=epoch)
                fid, vfid = self.calculate_vfid(netG, epoch, test_loader)
                print('[{}] fid:{:.4f}, vfid:{:.4f}'.format(epoch, fid, vfid))
                with open(output_score_filename, 'a') as f:
                    f.write('{},{},{}\n'.format(epoch, fid, vfid))

    def eval_fvd(self, imageloader, storyloader, testloader, stage=1):
        output_score_filename = os.path.join(self.save_dir, 'fvd_score.csv')
        save_dir = os.path.join(self.save_dir, 'epoch') # tmep file for save the iamges
        models = os.listdir(self.model_dir)
        with open(output_score_filename, 'a') as f:
            f.write('epoch,fvd\n')
        for epoch in range(121, 0, -1):
            if 'netG_epoch_{}.pth'.format(epoch) in models:
                print('Evaluating epoch {}'.format(epoch))
                netG = self.load_network_stageI(self.output_dir, load_ckpt=epoch)
                inference_samples(netG, testloader, save_dir)
                fvd_value = self.calculate_fvd(save_dir, epoch=epoch, num_of_video=272) #288)
                with open(output_score_filename, 'a') as f:
                    f.write('{},{}\n'.format(epoch, fvd_value))

    def inference(self, imageloader, storyloader, testloader, stage=1):
        netG = self.load_network_stageI(self.output_dir, load_ckpt=self.load_ckpt)
        inference_samples(netG, testloader, save_dir)

    def generate_story(self, netG, dataloader):
        from miscc.utils import images_to_numpy
        import PIL

        # netG, _, _ = self.load_network_stageI()
        # state_dict = torch.load(model_path,
        #                 map_location=lambda storage, loc: storage)
        # netG.load_state_dict(state_dict)

        origin_img_path = os.path.join(self.save_dir, 'original')
        generated_img_path = os.path.join(self.save_dir, 'generate')
        os.makedirs(origin_img_path, exist_ok=True)
        os.makedirs(generated_img_path, exist_ok=True)

        print('Generating Test Samples...')
        save_images, save_labels = [], []
        story_id = 0
        for batch in tqdm(dataloader):
            #print('Processing at ' + str(i))
            real_cpu = batch['images']
            motion_input = batch['description'][:, :, :cfg.TEXT.DIMENSION]
            content_input = batch['description'][:, :, :cfg.TEXT.DIMENSION]
            catelabel = batch['labels']
            real_imgs = Variable(real_cpu)
            motion_input = Variable(motion_input)
            content_input = Variable(content_input)
            if cfg.CUDA:
                real_imgs = real_imgs.cuda()            
                motion_input = motion_input.cuda()
                content_input = content_input.cuda()
                catelabel = catelabel.cuda()
            motion_input = torch.cat((motion_input, catelabel), 2)
            #content_input = torch.cat((content_input, catelabel), 2)
            _, fake_stories, _,_,_,_,_ = netG.sample_videos(motion_input, content_input)
            real_cpu = real_cpu.transpose(1, 2)
            fake_stories = fake_stories.transpose(1, 2)

            for (fake_story, real_story) in zip(fake_stories, real_cpu):
                origin_story_path = os.path.join(origin_img_path, str(story_id))
                os.makedirs(origin_story_path, exist_ok=True)
                generated_story_path = os.path.join(generated_img_path, str(story_id))
                os.makedirs(generated_story_path, exist_ok=True)

                for idx, (fake, real) in enumerate(zip(fake_story, real_story)):
                    fake_img = images_to_numpy(fake)
                    fake_img = PIL.Image.fromarray(fake_img)
                    fake_img.save(os.path.join(generated_story_path, str(idx)+'.png'))

                    real_img = images_to_numpy(real)
                    real_img = PIL.Image.fromarray(real_img)
                    real_img.save(os.path.join(origin_story_path, str(idx)+'.png'))
                
                story_id += 1

    def eval_fid2(self, testloader, video_transforms, image_transforms):
        from fid.fid_score import fid_score 
        output_score_filename = os.path.join(self.save_dir, 'fid_score2.csv')
        with open(output_score_filename, 'a') as f:
            f.write('epoch,fid,vfid\n')

        models = os.listdir(self.model_dir)

        for epoch in range(121, 0, -1):
            if 'netG_epoch_{}.pth'.format(epoch) in models:
                if os.path.exists(os.path.join(self.save_dir, 'original')):
                    shutil.rmtree(os.path.join(self.save_dir, 'original'))
                if os.path.exists(os.path.join(self.save_dir, 'generate')):
                    shutil.rmtree(os.path.join(self.save_dir, 'generate'))

                print('Evaluating epoch {}'.format(epoch))
                netG = self.load_network_stageI(self.output_dir, load_ckpt=epoch)
                with torch.no_grad():
                    self.generate_story(netG, testloader)
                ref_dataloader = FolderStoryDataset(os.path.join(self.save_dir, 'original'), video_transforms)
                gen_dataloader = FolderStoryDataset(os.path.join(self.save_dir, 'generate'), video_transforms)
                vfid = vfid_score(ref_dataloader,
                    gen_dataloader, cuda=True, normalize=True, r_cache=None)
                ref_dataloader = FolderImageDataset(os.path.join(self.save_dir, 'original'), image_transforms)
                gen_dataloader = FolderImageDataset(os.path.join(self.save_dir, 'generate'), image_transforms)

                fid = fid_score(ref_dataloader,
                        gen_dataloader, cuda=True, normalize=True, r_cache=None)
                with open(output_score_filename, 'a') as f:
                    f.write('{},{},{}\n'.format(epoch, fid, vfid))
