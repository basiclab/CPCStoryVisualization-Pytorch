from __future__ import print_function
from six.moves import range
from PIL import Image

import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import time
import pdb
import numpy as np
import torchfile

from miscc.config import cfg
from miscc.utils import mkdir_p
from miscc.utils import weights_init, count_param
from miscc.utils import save_story_results, save_model, save_test_samples, save_image_results
from miscc.utils import KL_loss
from miscc.utils import compute_discriminator_loss, compute_generator_loss
from story_fid import calculate_story_fid_given_activation
from story_fid import calculate_story_fid_given_activation, calculate_fid_given_activation
from shutil import copyfile
from story_fid_model import r2plus1d_18

from fid.vfid_score import fid_score as vfid_score
from fid.fid_score_v import fid_score
from fid.utils import StoryGANDataset, IgnoreLabelDataset

from torchvision import transforms
from tensorboardX import SummaryWriter
from inception import InceptionV3

from utils import StoryGANSSIMDataset
from ssim_score import ssim_score

from tqdm import tqdm

class GANTrainer(object):
    def __init__(self, output_dir, args, ratio=1.0):
        if cfg.TRAIN.FLAG:
            #output_dir = output_dir + '_r' + str(ratio) + '/'
            output_dir = "{}/".format(output_dir)
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            self.log_dir = os.path.join(output_dir, 'log')
            self.test_dir = os.path.join(output_dir, 'Test')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)
            mkdir_p(self.log_dir)
            mkdir_p(self.test_dir)
            if not os.path.exists(os.path.join(self.model_dir, 'model.py')):
                copyfile(args.cfg_file, output_dir + 'setting.yml')
                if cfg.CASCADE_MODEL:
                    copyfile('./cascade_model.py', output_dir + 'model.py')
                else:
                    copyfile('./model.py', output_dir + 'model.py')
                copyfile('./trainer.py', output_dir + 'trainer.py')

        self.video_len = cfg.VIDEO_LEN
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        s_gpus = cfg.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)
        self.imbatch_size = cfg.TRAIN.IM_BATCH_SIZE * self.num_gpus
        self.stbatch_size = cfg.TRAIN.ST_BATCH_SIZE * self.num_gpus
        self.ratio = ratio
        self.con_ckpt = args.continue_ckpt
        self.fid_eval = False
        torch.cuda.set_device(self.gpus[0])
        cudnn.benchmark = True
        # for fid
        self.inception_dim = 2048

        self._logger = SummaryWriter(self.log_dir)
    # ############# For training stageI GAN #############
    def load_network_stageI(self):
        if cfg.CASCADE_MODEL:
            from cascade_model import StoryGAN, STAGE1_D_IMG, STAGE1_D_STY_V2, STAGE1_D_SEG        
        else:
            from model import StoryGAN, STAGE1_D_IMG, STAGE1_D_STY_V2, STAGE1_D_SEG
        netG = StoryGAN(self.video_len)
        netG.apply(weights_init)
        netD_im = STAGE1_D_IMG()
        netD_im.apply(weights_init)
        netD_st = STAGE1_D_STY_V2()
        netD_st.apply(weights_init)
        #netD_se = STAGE1_D_STY_V2() # v1
        netD_se = None
        if cfg.SEGMENT_LEARNING:
            netD_se = STAGE1_D_SEG() # v2
            netD_se.apply(weights_init)

        netG_param_cnt, netD_im_param, netD_st_param = count_param(netG), count_param(netD_im), count_param(netD_st)
        total_params = netG_param_cnt + netD_im_param + netD_st_param
        if cfg.SEGMENT_LEARNING:
            netD_se_param_cnt = count_param(netD_se)
            total_params += netD_se_param_cnt
            print('Segment params : {} M'.format(netD_se_param_cnt//1e6))

        print('The total parameter is : {}M, netG:{}M, netD_im:{}M, netD_st:{}M'.format(total_params//1e6, netG_param_cnt//1e6,
            netD_im_param//1e6, netD_st_param//1e6))

        if cfg.NET_G != '':
            state_dict = \
                torch.load(cfg.NET_G,
                           map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load from: ', cfg.NET_G)
        if cfg.NET_D != '':
            state_dict = \
                torch.load(cfg.NET_D,
                           map_location=lambda storage, loc: storage)
            netD.load_state_dict(state_dict)
            print('Load from: ', cfg.NET_D)
        if self.con_ckpt:
            print('Continue training from epoch {}'.format(self.con_ckpt))
            path = '{}/netG_epoch_{}.pth'.format(self.model_dir, self.con_ckpt)
            netG.load_state_dict(torch.load(path))
            path = '{}/netD_im_epoch_last.pth'.format(self.model_dir)
            netD_im.load_state_dict(torch.load(path))
            path = '{}/netD_st_epoch_last.pth'.format(self.model_dir)
            netD_st.load_state_dict(torch.load(path))
            if cfg.SEGMENT_LEARNING:
                path = '{}/netD_se_epoch_last.pth'.format(self.model_dir)
                netD_se.load_state_dict(torch.load(path))

        if cfg.CUDA:
            netG.cuda()
            netD_im.cuda()
            netD_st.cuda()
            if cfg.SEGMENT_LEARNING:
                netD_se.cuda()

        return netG, netD_im, netD_st, netD_se


    def sample_real_image_batch(self):
        if self.imagedataset is None:
            self.imagedataset = enumerate(self.imageloader)
        batch_idx, batch = next(self.imagedataset)
        b = batch
        if cfg.CUDA:
            # Put each image into gpu
            for k, v in batch.items():
                if k == 'text':
                    continue
                else:
                    b[k] = v.cuda()

        if batch_idx == len(self.imageloader) - 1:
            self.imagedataset = enumerate(self.imageloader)
        return b

    def calculate_vfid(self, netG, epoch, testloader):
        netG.eval()
        with torch.no_grad():
            eval_modeldataset = StoryGANDataset(netG, len(testloader), testloader.dataset)
            vfid_value = vfid_score(IgnoreLabelDataset(testloader.dataset),
                eval_modeldataset, cuda=True, normalize=True, r_cache='.cache/seg_story_vfid_reference_score.npz'
            )
            fid_value = fid_score(IgnoreLabelDataset(testloader.dataset),
                    eval_modeldataset, cuda=True, normalize=True, r_cache='.cache/seg_story_fid_reference_score.npz'
                )
        netG.train()

        if self._logger:
            self._logger.add_scalar('Evaluation/vfid',  vfid_value,  epoch)
            self._logger.add_scalar('Evaluation/fid',  fid_value,  epoch)

    def calculate_ssim(self, netG, epoch, testloader):
        netG.eval()
        print('calculating SSIM')
        with torch.no_grad():
            eval_modeldataset = StoryGANSSIMDataset(netG, len(testloader), testloader.dataset)
            ssim_value = ssim_score(eval_modeldataset)
        netG.train()
        print('Epoch: {:d} ssim: {:.4f} ' .format(epoch, ssim_value) )
        if self._logger:
            self._logger.add_scalar('Evaluation/ssim', ssim_value, epoch)

    def train(self, imageloader, storyloader, testloader, stage=1):
        c_time = time.time()
        self.imageloader = imageloader
        self.imagedataset = None

        netG, netD_im, netD_st, netD_se = self.load_network_stageI()
        start = time.time()
        # Initial Labels
        im_real_labels = Variable(torch.FloatTensor(self.imbatch_size).fill_(1))
        im_fake_labels = Variable(torch.FloatTensor(self.imbatch_size).fill_(0))
        st_real_labels = Variable(torch.FloatTensor(self.stbatch_size).fill_(1))
        st_fake_labels = Variable(torch.FloatTensor(self.stbatch_size).fill_(0))
        if cfg.CUDA:
            im_real_labels, im_fake_labels = im_real_labels.cuda(), im_fake_labels.cuda()
            st_real_labels, st_fake_labels = st_real_labels.cuda(), st_fake_labels.cuda()

        use_segment = cfg.SEGMENT_LEARNING
        segment_weight = cfg.SEGMENT_RATIO
        image_weight = cfg.IMAGE_RATIO

        # Optimizer and Scheduler
        generator_lr = cfg.TRAIN.GENERATOR_LR
        discriminator_lr = cfg.TRAIN.DISCRIMINATOR_LR
        lr_decay_step = cfg.TRAIN.LR_DECAY_EPOCH

        im_optimizerD = optim.Adam(netD_im.parameters(), lr=cfg.TRAIN.DISCRIMINATOR_LR, betas=(0.5, 0.999))
        st_optimizerD = optim.Adam(netD_st.parameters(), lr=cfg.TRAIN.DISCRIMINATOR_LR, betas=(0.5, 0.999))
        if use_segment:
            se_optimizerD = optim.Adam(netD_se.parameters(), lr=cfg.TRAIN.DISCRIMINATOR_LR, betas=(0.5, 0.999))
        netG_para = []
        for p in netG.parameters():
            if p.requires_grad:
                netG_para.append(p)
        optimizerG = optim.Adam(netG_para, lr=cfg.TRAIN.GENERATOR_LR, betas=(0.5, 0.999))

        mse_loss = nn.MSELoss()

        scheduler_imD = ReduceLROnPlateau(im_optimizerD, 'min', verbose=True, factor=0.5, min_lr=1e-7, patience=0)
        scheduler_stD = ReduceLROnPlateau(st_optimizerD, 'min', verbose=True, factor=0.5, min_lr=1e-7, patience=0)
        if use_segment:
            scheduler_seD = ReduceLROnPlateau(se_optimizerD, 'min', verbose=True, factor=0.5, min_lr=1e-7, patience=0)
        scheduler_G = ReduceLROnPlateau(optimizerG, 'min', verbose=True, factor=0.5, min_lr=1e-7, patience=0)
        count = 0

        # Start training
        if not self.con_ckpt:
            start_epoch = 0
        else:
            start_epoch = int(self.con_ckpt)
        # self.calculate_vfid(netG, 0, testloader)

        print('LR DECAY EPOCH: {}'.format(lr_decay_step))
        for epoch in range(start_epoch, self.max_epoch):
            l = self.ratio * (2. / (1. + np.exp(-10. * epoch)) - 1)
            start_t = time.time()

            # Adjust lr
            num_step = len(storyloader)
            stats = {}

            with tqdm(total=len(storyloader), dynamic_ncols=True) as pbar:
                for i, data in enumerate(storyloader):
                    ######################################################
                    # (1) Prepare training data
                    ######################################################
                    im_batch = self.sample_real_image_batch()
                    st_batch = data
                    im_real_cpu = im_batch['images']
                    im_motion_input = im_batch['description'][:, :cfg.TEXT.DIMENSION] # description vector and arrtibute (60, 356)
                    im_content_input = im_batch['content'][:, :, :cfg.TEXT.DIMENSION] # description vector and attribute for every story (60,5,356)
                    im_real_imgs = Variable(im_real_cpu)
                    im_motion_input = Variable(im_motion_input)
                    im_content_input = Variable(im_content_input)
                    im_labels = Variable(im_batch['labels'])

                    st_real_cpu = st_batch['images']
                    st_motion_input = st_batch['description'][:, :, :cfg.TEXT.DIMENSION] #(12,5,356)
                    st_content_input = st_batch['description'][:, :, :cfg.TEXT.DIMENSION] # (12,5,356)
                    st_texts = None
                    if 'text' in st_batch:
                        st_texts = st_batch['text']
                    st_real_imgs = Variable(st_real_cpu)
                    st_motion_input = Variable(st_motion_input)
                    st_content_input = Variable(st_content_input)
                    st_labels = Variable(st_batch['labels']) # (12,5,9)
                    if use_segment:
                        se_real_cpu = im_batch['images_seg']
                        se_real_imgs = Variable(se_real_cpu)

                    if cfg.CUDA:
                        st_real_imgs = st_real_imgs.cuda() # (12,3,5,64,64)
                        im_real_imgs = im_real_imgs.cuda()
                        st_motion_input = st_motion_input.cuda()
                        im_motion_input = im_motion_input.cuda()
                        st_content_input = st_content_input.cuda()
                        im_content_input = im_content_input.cuda()
                        im_labels = im_labels.cuda()
                        st_labels = st_labels.cuda()
                        if use_segment:
                            se_real_imgs = se_real_imgs.cuda()
                    im_motion_input = torch.cat((im_motion_input, im_labels), 1) # 356+9=365 (60,365)
                    st_motion_input = torch.cat((st_motion_input, st_labels), 2) # (12,5,365)

                    #######################################################
                    # (2) Generate fake stories and images
                    ######################################################
                    # print(st_motion_input.shape, im_motion_input.shape)

                    with torch.no_grad():
                        _, st_fake, m_mu, m_logvar, c_mu, c_logvar, _ = \
                            netG.sample_videos(st_motion_input, st_content_input) # m_mu (60,365), c_mu (12,124)

                        _, im_fake, im_mu, im_logvar, cim_mu, cim_logvar, se_fake = \
                            netG.sample_images(im_motion_input, im_content_input, seg=True) # im_mu (60,489), cim_mu (60,124)


                    characters_mu = (st_labels.mean(1)>0).type(torch.FloatTensor).cuda() # which character exists in the full story (5 descriptions)
                    st_mu = torch.cat((c_mu, st_motion_input[:,:, :cfg.TEXT.DIMENSION].mean(1).squeeze(), characters_mu), 1)
                    #  124 + 356 + 9 = 489 (12,489), get character info form whole story

                    im_mu = torch.cat((im_motion_input, cim_mu), 1)
                    # (60,489)
                    ############################
                    # (3) Update D network
                    ###########################

                    netD_im.zero_grad()
                    netD_st.zero_grad()
                    se_accD = 0
                    if use_segment:
                        netD_se.zero_grad()
                        se_errD, se_errD_real, se_errD_wrong, se_errD_fake, se_accD, _ = \
                            compute_discriminator_loss(netD_se, se_real_imgs, se_fake,
                                                im_real_labels, im_fake_labels, im_labels,
                                                im_mu, self.gpus)

                    im_errD, im_errD_real, im_errD_wrong, im_errD_fake, im_accD, _ = \
                        compute_discriminator_loss(netD_im, im_real_imgs, im_fake,
                                               im_real_labels, im_fake_labels, im_labels,
                                               im_mu, self.gpus)

                    st_errD, st_errD_real, st_errD_wrong, st_errD_fake, _, order_consistency  = \
                        compute_discriminator_loss(netD_st, st_real_imgs, st_fake,
                                               st_real_labels, st_fake_labels, st_labels,
                                               st_mu, self.gpus)

                    if use_segment:
                        se_errD.backward()
                        se_optimizerD.step()
                        stats.update({
                            'seg_D/loss': se_errD.data,
                            'seg_D/real': se_errD_real,
                            'seg_D/fake': se_errD_fake,
                        })

                    im_errD.backward()
                    st_errD.backward()

                    im_optimizerD.step()
                    st_optimizerD.step()

                    stats.update({
                        'img_D/loss': im_errD.data,
                        'img_D/real': im_errD_real,
                        'img_D/fake': im_errD_fake,
                        'Accuracy/im_D': im_accD,
                        'Accuracy/se_D': se_accD,
                    })

                    step = i+num_step*epoch
                    self._logger.add_scalar('st_D/loss', st_errD.data, step)
                    self._logger.add_scalar('st_D/real', st_errD_real, step)
                    self._logger.add_scalar('st_D/fake', st_errD_fake, step)
                    self._logger.add_scalar('st_D/order', order_consistency, step)

                    ############################
                    # (2) Update G network
                    ###########################
                    netG.zero_grad()
                    video_latents, st_fake, m_mu, m_logvar, c_mu, c_logvar, _ = netG.sample_videos(st_motion_input, st_content_input)
                    image_latents, im_fake, im_mu, im_logvar, cim_mu, cim_logvar, se_fake = netG.sample_images(im_motion_input, im_content_input,
                        seg=use_segment)
                    encoder_decoder_loss = 0
                    if video_latents is not None:
                        ((h_seg1, h_seg2, h_seg3, h_seg4), (g_seg1, g_seg2, g_seg3, g_seg4)) = video_latents

                        video_latent_loss = mse_loss(g_seg1, h_seg1) + mse_loss(g_seg2, h_seg2 ) + mse_loss(g_seg3, h_seg3) + mse_loss(g_seg4, h_seg4)
                        ((h_seg1, h_seg2, h_seg3, h_seg4), (g_seg1, g_seg2, g_seg3, g_seg4)) = image_latents
                        image_latent_loss = mse_loss(g_seg1, h_seg1) + mse_loss(g_seg2, h_seg2 ) + mse_loss(g_seg3, h_seg3) + mse_loss(g_seg4, h_seg4)
                        encoder_decoder_loss = ( image_latent_loss + video_latent_loss ) / 2

                        reconstruct_img = netG.train_autoencoder(se_real_imgs)
                        reconstruct_fake = netG.train_autoencoder(se_fake)
                        reconstruct_loss = (mse_loss(reconstruct_img, se_real_imgs) + mse_loss(reconstruct_fake, se_fake)) / 2.0

                        self._logger.add_scalar('G/image_vae_loss', image_latent_loss.data, step)
                        self._logger.add_scalar('G/video_vae_loss', video_latent_loss.data, step)
                        self._logger.add_scalar('G/reconstruct_loss', reconstruct_loss.data, step)

                    characters_mu = (st_labels.mean(1)>0).type(torch.FloatTensor).cuda()
                    st_mu = torch.cat((c_mu, st_motion_input[:,:, :cfg.TEXT.DIMENSION].mean(1).squeeze(), characters_mu), 1)

                    im_mu = torch.cat((im_motion_input, cim_mu), 1)
                    se_errG, se_errG, se_accG = 0, 0, 0
                    if use_segment:
                        se_errG, se_accG, _ = compute_generator_loss(netD_se, se_fake, se_real_imgs,
                                                    im_real_labels, im_labels, im_mu, self.gpus)

                    im_errG, im_accG, _ = compute_generator_loss(netD_im, im_fake, im_real_imgs,
                                                im_real_labels, im_labels, im_mu, self.gpus)

                    st_errG, st_accG, G_consistency  = compute_generator_loss(netD_st, st_fake, st_real_imgs,
                                                st_real_labels, st_labels, st_mu, self.gpus)
                    ######
                    # Sample Image Loss and Sample Video Loss
                    im_kl_loss = KL_loss(cim_mu, cim_logvar)
                    st_kl_loss = KL_loss(c_mu, c_logvar)

                    errG =  im_errG + self.ratio * ( image_weight*st_errG + se_errG*segment_weight) # for record
                    kl_loss = im_kl_loss + self.ratio * st_kl_loss # for record

                    # Total Loss
                    errG_total = im_errG + im_kl_loss * cfg.TRAIN.COEFF.KL \
                        + self.ratio * (se_errG*segment_weight + st_errG*image_weight + st_kl_loss * cfg.TRAIN.COEFF.KL)

                    if video_latents is not None:
                        errG_total += ( video_latent_loss +  reconstruct_loss )* cfg.RECONSTRUCT_LOSS

                    errG_total.backward()
                    optimizerG.step()
                    stats.update({
                        'G/loss': errG_total.data,
                        'G/im_KL': im_kl_loss.data,
                        'G/st_KL': st_kl_loss.data,
                        'G/KL': kl_loss.data,
                        'G/consistency': G_consistency,
                        'Accuracy/im_G': im_accG,
                        'Accuracy/se_G': se_accG,
                        'Accuracy/st_G': st_accG,
                        'G/gan_loss': errG.data,
                    })

                    count = count + 1
                    pbar.update(1)

                    if i % 20 == 0:
                        step = i+num_step*epoch
                        for key, value in stats.items():
                            self._logger.add_scalar(key, value, step)

            with torch.no_grad():
                lr_fake, fake,_,_,_, _, se_fake = netG.sample_videos(st_motion_input, st_content_input, seg=use_segment)
                st_result = save_story_results(st_real_cpu, fake, st_texts, epoch, self.image_dir, i)
                if use_segment and se_fake is not None:
                    se_result = save_image_results(None, se_fake)
            self._logger.add_image("pororo", st_result.transpose(2,0,1)/255, epoch)
            if use_segment:
                self._logger.add_image("segment", se_result.transpose(2,0,1)/255, epoch)

            # Adjust lr
            if epoch % lr_decay_step == 0 and epoch > 0:
                generator_lr *= 0.5
                for param_group in optimizerG.param_groups:
                    param_group['lr'] = generator_lr
                discriminator_lr *= 0.5
                for param_group in st_optimizerD.param_groups:
                    param_group['lr'] = discriminator_lr
                for param_group in im_optimizerD.param_groups:
                    param_group['lr'] = discriminator_lr
                lr_decay_step *= 2

            g_lr, im_lr, st_lr = 0, 0, 0
            for param_group in optimizerG.param_groups:
                g_lr = param_group['lr']
            for param_group in st_optimizerD.param_groups:
                st_lr = param_group['lr']
            for param_group in im_optimizerD.param_groups:
                im_lr = param_group['lr']
            self._logger.add_scalar('learning/generator',g_lr, epoch)
            self._logger.add_scalar('learning/st_discriminator', st_lr, epoch)
            self._logger.add_scalar('learning/im_discriminator', im_lr, epoch)

            if cfg.EVALUATE_FID_SCORE:
                self.calculate_vfid(netG, epoch, testloader)

            #self.calculate_ssim(netG, epoch, testloader)
            time_mins = int((time.time() - c_time)/60)
            time_hours = int(time_mins / 60)
            epoch_mins = int((time.time()-start_t)/60)
            epoch_hours = int(epoch_mins / 60)

            print("----[{}/{}]Epoch time:{} hours {} mins, Total time:{} hours----".format(epoch, self.max_epoch, epoch_hours, epoch_mins, time_hours))
            #print('[{}/{}][{}/{}] LossG:{:.4f} LossD_se:{:.4f} LossD_im:{:.4f} LossD_st:{:.4f}'\
            #              .format(epoch, self.max_epoch, i, num_step, errG_total.data, se_errD.data, im_errD.data, st_errD.data))

            if epoch % self.snapshot_interval == 0:
                save_model(netG, netD_im, netD_st, netD_se, epoch, self.model_dir)
                #save_test_samples(netG, testloader, self.test_dir)
        save_model(netG, netD_im, netD_st, netD_se, self.max_epoch, self.model_dir)
