import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
from torch.nn.utils import spectral_norm
from torchvision.models.video.resnet import r2plus1d_18
from miscc.config import cfg
from torch.autograd import Variable
import numpy as np
import pdb
if torch.cuda.is_available():
    T = torch.cuda
else:
    T = torch

def conv3x3(in_planes, out_planes, stride=1, use_spectral_norm=False):
    "3x3 convolution with padding"
    if use_spectral_norm:
        return spectral_norm(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False))
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    #print("in_planes: {}, out_planes: {}".format(in_planes, out_planes))
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        #nn.functional.interpolate(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(True))
    return block

def downBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(True))
    return block


class CA_NET(nn.Module):
    # some code is modified from vae examples
    # (https://github.com/pytorch/examples/blob/master/vae/main.py)
    def __init__(self):
        super(CA_NET, self).__init__()
        self.t_dim = cfg.TEXT.DIMENSION * cfg.VIDEO_LEN
        self.c_dim = cfg.GAN.CONDITION_DIM
        self.fc = nn.Linear(self.t_dim, self.c_dim * 2, bias=True)
        self.relu = nn.ReLU()

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.c_dim]
        logvar = x[:, self.c_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if cfg.CUDA:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar


class D_GET_LOGITS(nn.Module):
    def __init__(self, ndf, nef, bcondition=True):
        super(D_GET_LOGITS, self).__init__()
        self.df_dim = ndf
        self.ef_dim = nef
        self.bcondition = bcondition
        if bcondition:
            self.outlogits = nn.Sequential(
                conv3x3(ndf * 8 + nef, ndf * 8, use_spectral_norm=True),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4)),
                nn.Sigmoid())
        else:
            self.outlogits = nn.Sequential(
                spectral_norm(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4)),
                nn.Sigmoid())

    def forward(self, h_code, c_code=None):
        # conditioning output   
        if self.bcondition and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            # state size (ngf+egf) x 4 x 4
            h_c_code = torch.cat((h_code, c_code), 1)
        else:
            h_c_code = h_code

        output = self.outlogits(h_c_code)
        return output.view(-1)

class R2Plus1dStem(nn.Sequential):
    """R(2+1)D stem is different than the default one as it uses separated 3D convolution
    """
    def __init__(self):
        super(R2Plus1dStem, self).__init__(
            spectral_norm(nn.Conv3d(3, 45, kernel_size=(1, 7, 7),
                      stride=(1, 2, 2), padding=(0, 3, 3),
                      bias=False)),
            nn.BatchNorm3d(45),
            nn.ReLU(inplace=True),
            spectral_norm(nn.Conv3d(45, 64, kernel_size=(1, 1, 1),
                      stride=(1, 1, 1), padding=(1, 0, 0),
                      bias=False)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))

class BasicBlock(nn.Module):

    __constants__ = ['downsample']
    expansion = 1

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            conv_builder(inplanes, planes, midplanes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes),
            nn.BatchNorm3d(planes)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class VideoEncoder(nn.Module):
    def __init__(self):
        super(VideoEncoder, self).__init__()
        video_resnet = r2plus1d_18(pretrained=False, progress=True)
        padding= 1
        block = [
            R2Plus1dStem(),
            spectral_norm(nn.Conv3d(64, 128, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, padding, padding)
                ,bias=False)),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv3d(128, 128, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(padding, 0, 0),
                            bias=False)),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv3d(128, 128, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, padding, padding), 
                bias=False)),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv3d(128, 256, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(padding, 0, 0),
                bias=False)),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv3d(256, 256, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, padding, padding), 
                bias=False)),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv3d(256, 512, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(padding, 0, 0),
                bias=False)),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv3d(512, 512, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, padding, padding), 
                bias=False)),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv3d(512, 512, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(padding, 0, 0), 
                bias=False)),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2),
        ]
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.story_encoder = nn.Sequential(*block)
        self.detector = nn.Sequential(
            spectral_norm(nn.Linear(512, 128)),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            spectral_norm(nn.Linear(128, 1)),
        )

    def forward(self, story):
        '''
            story: B x T X C X W X H
            B: batch size, N : number of story, T story length
            C: image channel, WxH width and height
        '''
        B = story.shape[0]
        latents = self.story_encoder(story)
        latents = self.pool(latents)
        latents = latents.view(B, -1)
        return self.detector(latents)


# ############# Networks for stageI GAN #############
class StoryGAN(nn.Module):
    def __init__(self, video_len):
        super(StoryGAN, self).__init__()
        print('Cascade model')
        self.batch_size = cfg.TRAIN.IM_BATCH_SIZE
        self.gf_dim = cfg.GAN.GF_DIM * 8 # 128*8=1024
        self.gf_dim_seg = cfg.GAN.GF_SEG_DIM #512 #48
        self.motion_dim = cfg.TEXT.DIMENSION + cfg.LABEL_NUM # (356+9=365)
        self.content_dim = cfg.GAN.CONDITION_DIM # encoded text dim (124)
        self.noise_dim = cfg.GAN.Z_DIM  # noise (100)
        print(self.motion_dim, self.gf_dim_seg, self.gf_dim, self.content_dim)
        print(self.content_dim)

        self.recurrent = nn.GRUCell(self.noise_dim + self.motion_dim, self.motion_dim) # (465,365)
        self.mocornn = nn.GRUCell(self.motion_dim, self.content_dim) # (365,124)
        self.video_len = video_len
        self.n_channels = 3
        self.filter_num = 3
        self.filter_size = 21
        self.image_size = 124
        self.out_num = 1
        # for segment image v1
        self.use_segment = cfg.SEGMENT_LEARNING

        self.segment_size = 4*2*2*2*2 # inital size is 4, upsample 4 times = 64
        self.segment_flat_size = 3*self.segment_size**2 # 12288
        # v2
        self.aux_size = 5
        self.fix_input = 0.1*torch.tensor(range(self.aux_size)).float().cuda()

        self.define_module()

    def define_module(self):
        from layers import DynamicFilterLayer1D as DynamicFilterLayer
        ninput = self.motion_dim + self.content_dim + self.image_size # (365+124+124=613)
        ngf = self.gf_dim # 128*8=1024
        
        self.ca_net = CA_NET()
        # -> ngf x 4 x 4
        
        self.filter_net = nn.Sequential(
            nn.Linear(self.content_dim, self.filter_size * self.filter_num * self.out_num),
            nn.BatchNorm1d(self.filter_size * self.filter_num * self.out_num))

        self.image_net = nn.Sequential(
            nn.Linear(self.motion_dim, self.image_size * self.filter_num),
            nn.BatchNorm1d(self.image_size * self.filter_num),
            nn.Tanh())
        
        # For generate final image
        self.fc = nn.Sequential(
            nn.Linear(ninput, ngf * 4 * 4, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4),
            nn.ReLU(True))
        self.upsample1 = upBlock(ngf, ngf//2)
        # -> ngf/4 x 16 x 16
        self.upsample2 = upBlock(ngf//2, ngf//4) 
        # -> ngf/8 x 32 x 32
        self.upsample3 = upBlock(ngf//4, ngf//8)
        # -> ngf/16 x 64 x 64
        self.upsample4 = upBlock(ngf//8, ngf//16)
        # -> 3 x 64 x 64
        self.img = nn.Sequential(
            conv3x3(ngf // 16, 3),
            nn.Tanh())
        if self.use_segment:
            ngf_seg = self.gf_dim_seg

            self.seg_c = conv3x3(ngf_seg, ngf)     # 2048, 256
            self.seg_c1 = conv3x3(ngf_seg//2, ngf//2) # 1024, 128
            # self.seg_c2 = conv3x3(ngf_seg//4, ngf//4)
            # self.seg_c3 = conv3x3(ngf_seg//8, ngf//8)
            # self.seg_c4 = conv3x3(ngf_seg//16, ngf//16)

            # For generate seg and img v4 and v5 and v6
            self.fc_seg = nn.Sequential(
                nn.Linear(ninput, ngf_seg * 4 * 4, bias=False),
                nn.BatchNorm1d(ngf_seg * 4 * 4),
                nn.ReLU(True))
            # ngf x 4 x 4 -> ngf/2 x 8 x 8
            self.upsample1_seg = upBlock(ngf_seg, ngf_seg // 2)
            # -> ngf/4 x 16 x 16
            self.upsample2_seg = upBlock(ngf_seg // 2, ngf_seg // 4)
            # -> ngf/8 x 32 x 32
            self.upsample3_seg = upBlock(ngf_seg // 4, ngf_seg // 8)
            # -> ngf/16 x 64 x 64
            self.upsample4_seg = upBlock(ngf_seg // 8, ngf_seg // 16)
            # -> 1 x 64 x 64
            self.img_seg = nn.Sequential(
                conv3x3(ngf_seg // 16, 1),
                nn.Tanh())
            self.presample = nn.Sequential(
                conv3x3(1, ngf_seg // 16),
                nn.BatchNorm2d(ngf_seg // 16),
                nn.ReLU(),
            )
            self.downsample1_seg = downBlock( ngf_seg // 16, ngf_seg // 8 )
            self.downsample2_seg = downBlock( ngf_seg // 8, ngf_seg // 4 )
            self.downsample3_seg = downBlock( ngf_seg // 4, ngf_seg // 2 )
            self.downsample4_seg = downBlock( ngf_seg // 2, ngf_seg )


        self.m_net = nn.Sequential(
            nn.Linear(self.motion_dim, self.motion_dim),
            nn.BatchNorm1d(self.motion_dim))
        self.c_net = nn.Sequential(
            nn.Linear(self.content_dim, self.content_dim),
            nn.BatchNorm1d(self.content_dim))

        self.dfn_layer = DynamicFilterLayer(self.filter_size, 
            pad = self.filter_size//2)

    def get_iteration_input(self, motion_input):
        num_samples = motion_input.shape[0]
        noise = T.FloatTensor(num_samples, self.noise_dim).normal_(0,1)
        return torch.cat((noise, motion_input), dim = 1)

    def get_gru_initial_state(self, num_samples):
        return Variable(T.FloatTensor(num_samples, self.motion_dim).normal_(0, 1))

    def sample_z_motion(self, motion_input, video_len=None):
        video_len = video_len if video_len is not None else self.video_len
        num_samples = motion_input.shape[0]
        h_t = [self.m_net(self.get_gru_initial_state(num_samples))]
        
        for frame_num in range(video_len):
            if len(motion_input.shape) == 2:
                e_t = self.get_iteration_input(motion_input)
            else:
                e_t = self.get_iteration_input(motion_input[:,frame_num,:])
            h_t.append(self.recurrent(e_t, h_t[-1]))
        z_m_t = [h_k.view(-1, 1, self.motion_dim) for h_k in h_t]
        z_motion = torch.cat(z_m_t[1:], dim=1).view(-1, self.motion_dim)
        return z_motion

    def motion_content_rnn(self, motion_input, content_input):
        video_len = 1 if len(motion_input.shape) == 2 else self.video_len
        h_t = [self.c_net(content_input)]
        if len(motion_input.shape) == 2:
            motion_input = motion_input.unsqueeze(1)
        for frame_num in range(video_len):
            h_t.append(self.mocornn(motion_input[:,frame_num, :], h_t[-1]))
        
        c_m_t = [h_k.view(-1, 1, self.content_dim) for h_k in h_t]
        mocornn_co = torch.cat(c_m_t[1:], dim=1).view(-1, self.content_dim)
        return mocornn_co

    def sample_videos(self, motion_input, content_input, seg=False):  

        ###
        # motion_input:  batch_size, video_len, 365
        # content_input: batch_size, video_len, 356
        ###
        bs, video_len  = motion_input.shape[0], motion_input.shape[1]
        num_img = bs * video_len
        content_input = content_input.view(-1, cfg.VIDEO_LEN * content_input.shape[2])
        if content_input.shape[0] > 1:
            content_input = torch.squeeze(content_input)
        r_code, r_mu, r_logvar = self.ca_net(content_input) ## h0 
        #c_code = r_code.repeat(self.video_len, 1).view(-1, r_code.shape[1])
        c_mu = r_mu.repeat(self.video_len, 1).view(-1, r_mu.shape[1])
        #c_logvar = r_logvar.repeat(self.video_len, 1).view(-1, r_logvar.shape[1])

        crnn_code = self.motion_content_rnn(motion_input, r_code) ## i_t = GRU(s_t)
        temp = motion_input.view(-1, motion_input.shape[2])
        m_code, m_mu, m_logvar = temp, temp, temp #self.ca_net(temp)
        m_code = m_code.view(motion_input.shape[0], self.video_len, self.motion_dim)
        zm_code = self.sample_z_motion(m_code, self.video_len) ## *

        # one
        zmc_code = torch.cat((zm_code, c_mu), dim = 1)
        # two
        m_image = self.image_net(m_code.view(-1, m_code.shape[2])) ## linearly transform motion(365) to image(372)
        m_image = m_image.view(-1, self.filter_num, self.image_size)
        c_filter = self.filter_net(crnn_code) ## Filter(i_t)
        c_filter = c_filter.view(-1, self.out_num, self.filter_num, self.filter_size)
        mc_image = self.dfn_layer([m_image, c_filter]) ## *
        zmc_all_ = torch.cat((zmc_code, mc_image.squeeze(1)), dim = 1)
        zmc_img = self.fc(zmc_all_).view(-1, self.gf_dim, 4, 4)

        if self.use_segment:
            zmc_seg = self.fc_seg(zmc_all_).view(-1, self.gf_dim_seg, 4, 4)
            # print(zmc_seg.shape)
            h_seg1 = self.upsample1_seg(zmc_seg) #;print(h_seg1.shape)
            h_seg2 = self.upsample2_seg(h_seg1) #;print(h_seg2.shape)
            h_seg3 = self.upsample3_seg(h_seg2) #;print(h_seg3.shape)
            h_seg4 = self.upsample4_seg(h_seg3) #;print(h_seg4.shape)

            # generate seg
            segm_video = self.img_seg(h_seg4)

            zmc_latent = self.presample(segm_video) #;print(zmc_latent.shape)
            g_seg4 = self.downsample1_seg(zmc_latent) #;print(g_seg4.shape)
            g_seg3 = self.downsample2_seg(g_seg4) #;print(g_seg3.shape)
            g_seg2 = self.downsample3_seg(g_seg3) #;print(g_seg2.shape)
            g_seg1 = self.downsample4_seg(g_seg2) #;print(g_seg1.shape)


            zmc_img = self.seg_c(g_seg1) * zmc_img + zmc_img
            

            h_img = self.upsample1(zmc_img)
            h_img = self.seg_c1(g_seg2) * h_img + h_img # batch_size*video_len, 1024, 8, 8

            h_img = self.upsample2(h_img)
            # h_img = self.seg_c2(g_seg3) * h_img + h_img

            h_img = self.upsample3(h_img)
            # h_img = self.seg_c3(g_seg4) * h_img + h_img

            h_img = self.upsample4(h_img)
            # h_img = self.seg_c4(zmc_latent) * h_img + h_img
            segm_temp = segm_video.view(-1, self.video_len, 1, self.segment_size, self.segment_size)
            segm_temp = segm_temp.permute(0, 2, 1, 3, 4)
            # generate video
            fake_video = self.img(h_img)
            fake_video = fake_video.view(-1, self.video_len, self.n_channels, self.segment_size, self.segment_size)
            fake_video = fake_video.permute(0, 2, 1, 3, 4)

            if seg==True:
                return ((zmc_seg, h_seg1, h_seg2, h_seg3), (g_seg1, g_seg2, g_seg3, g_seg4)), \
                    fake_video,  m_mu, m_logvar, r_mu, r_logvar, segm_video # m_mu(60,365), m_logvar(60,365), r_mu(12,124), r_logvar(12,124)
            else:
                return ((zmc_seg, h_seg1, h_seg2, h_seg3), (g_seg1, g_seg2, g_seg3, g_seg4)), \
                    fake_video,  m_mu, m_logvar, r_mu, r_logvar, None # m_mu(60,365), m_logvar(60,365), r_mu(12,124), r_logvar(12,124)
        else:
            h_code = self.upsample1(zmc_img) # h_code: batch_size*video_len, 1024, 8, 8 *
            h_code = self.upsample2(h_code)  # h_code: batch_size*video_len, 512, 16, 16 *
            h_code = self.upsample3(h_code)  # h_code: batch_size*video_len, 256, 32, 32 *
            h_code = self.upsample4(h_code)  # h_code: batch_size*video_len=60, 128, 64, 64 *
            # state size 3 x 64 x 64
            h = self.img(h_code) ## *
            fake_video = h.view( int(h.size(0)/self.video_len), self.video_len, self.n_channels, h.size(3), h.size(3)) # 12, 5, 3, 64, 64
            fake_video = fake_video.permute(0, 2, 1, 3, 4) # 12, 3, 5, 64, 64
            #pdb.set_trace()
            return None, fake_video,  m_mu, m_logvar, r_mu, r_logvar, None # m_mu(60,365), m_logvar(60,365), r_mu(12,124), r_logvar(12,124)


    def sample_images(self, motion_input, content_input, seg=False):
        ### Adding segmenation result ###
        bs, video_len  = motion_input.shape[0], motion_input.shape[1]
        num_img = bs         
        m_code, m_mu, m_logvar = motion_input, motion_input, motion_input
        content_input = content_input.reshape(-1, cfg.VIDEO_LEN * content_input.shape[2])
        c_code, c_mu, c_logvar = self.ca_net(content_input) ## h0
        crnn_code = self.motion_content_rnn(motion_input, c_mu) ## GRU
        zm_code = self.sample_z_motion(m_code, 1) ## Text2Gist
        # one
        zmc_code = torch.cat((zm_code, c_mu), dim = 1) # (60,365 ; 60,124)->(60,489)
        # two
        m_image = self.image_net(m_code) ## * 
        m_image = m_image.view(-1, self.filter_num, self.image_size) #(60,3,124)
        c_filter = self.filter_net(crnn_code) ## *
        c_filter = c_filter.view(-1, self.out_num, self.filter_num, self.filter_size)
        mc_image = self.dfn_layer([m_image, c_filter]) ## * #(60,1,124)
        zmc_all_ = torch.cat((zmc_code, mc_image.squeeze(1)), dim = 1) # (60,613)

        zmc_img = self.fc(zmc_all_).view(-1, self.gf_dim, 4, 4)
        if self.use_segment:
            zmc_seg = self.fc_seg(zmc_all_).view(-1, self.gf_dim_seg, 4, 4)
            # print(zmc_seg.shape)
            h_seg1 = self.upsample1_seg(zmc_seg) #;print(h_seg1.shape)
            h_seg2 = self.upsample2_seg(h_seg1) #;print(h_seg2.shape)
            h_seg3 = self.upsample3_seg(h_seg2) #;print(h_seg3.shape)
            h_seg4 = self.upsample4_seg(h_seg3) #;print(h_seg4.shape)

            # generate seg
            segm_img = self.img_seg(h_seg4)

            zmc_latent = self.presample(segm_img) #;print(zmc_latent.shape)
            g_seg4 = self.downsample1_seg(zmc_latent) #;print(g_seg4.shape)
            g_seg3 = self.downsample2_seg(g_seg4) #;print(g_seg3.shape)
            g_seg2 = self.downsample3_seg(g_seg3) #;print(g_seg2.shape)
            g_seg1 = self.downsample4_seg(g_seg2) #;print(g_seg1.shape)


            zmc_img = self.seg_c(g_seg1) * zmc_img + zmc_img

            h_img = self.upsample1(zmc_img)
            h_img = self.seg_c1(g_seg2) * h_img + h_img # batch_size*video_len, 1024, 8, 8

            h_img = self.upsample2(h_img)
            # h_img = self.seg_c2(g_seg3) * h_img + h_img

            h_img = self.upsample3(h_img)
            # h_img = self.seg_c3(g_seg4) * h_img + h_img
            h_img = self.upsample4(h_img)

            # generatte video
            fake_img = self.img(h_img)
            fake_img = fake_img.view(-1, self.n_channels, self.segment_size, self.segment_size)

            if seg==True:
                return ((zmc_seg, h_seg1, h_seg2, h_seg3), (g_seg1, g_seg2, g_seg3, g_seg4)), \
                    fake_img, m_mu, m_logvar, c_mu, c_logvar, segm_img
            else:
                return ((zmc_seg, h_seg1, h_seg2, h_seg3), (g_seg1, g_seg2, g_seg3, g_seg4)), \
                         fake_img, m_mu, m_logvar, c_mu, c_logvar, None
        else:
            h_code = self.upsample1(zmc_img) ## *
            h_code = self.upsample2(h_code)  ## *
            h_code = self.upsample3(h_code)  ## *
            h_code = self.upsample4(h_code)  ## *
            # state size 3 x 64 x 64
            fake_img = self.img(h_code)
            return None, fake_img, m_mu, m_logvar, c_mu, c_logvar, None

    def train_autoencoder(self, real_segments):
        zmc_latent = self.presample(real_segments) #;print(zmc_latent.shape)
        g_seg4 = self.downsample1_seg(zmc_latent) #;print(g_seg4.shape)
        g_seg3 = self.downsample2_seg(g_seg4) #;print(g_seg3.shape)
        g_seg2 = self.downsample3_seg(g_seg3) #;print(g_seg2.shape)
        g_seg1 = self.downsample4_seg(g_seg2) #;print(g_seg1.shape)

        h_seg1 = self.upsample1_seg(g_seg1) #;print(h_seg1.shape)
        h_seg2 = self.upsample2_seg(h_seg1) #;print(h_seg2.shape)
        h_seg3 = self.upsample3_seg(h_seg2) #;print(h_seg3.shape)
        h_seg4 = self.upsample4_seg(h_seg3) #;print(h_seg4.shape)
        segm_img = self.img_seg(h_seg4)
        return segm_img #(( h_seg1, h_seg2, h_seg3, h_seg4), (g_seg1, g_seg2, g_seg3, g_seg4))


class STAGE1_D_IMG(nn.Module):
    def __init__(self, use_categories = True):
        super(STAGE1_D_IMG, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.CONDITION_DIM
        self.text_dim = cfg.TEXT.DIMENSION
        self.label_num = cfg.LABEL_NUM
        self.define_module(use_categories)

    def define_module(self, use_categories):
        ndf, nef = self.df_dim, self.ef_dim
        self.encode_img = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*2) x 16 x 16
            spectral_norm(nn.Conv2d(ndf*2, ndf * 4, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*4) x 8 x 8
            spectral_norm(nn.Conv2d(ndf*4, ndf * 8, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 8),
            # state size (ndf * 8) x 4 x 4)
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.seq_consisten_model = None
        self.get_cond_logits = D_GET_LOGITS(ndf, nef + self.text_dim + self.label_num)
        self.get_uncond_logits = None

        if use_categories:
            self.cate_classify = nn.Conv2d(ndf * 8, self.label_num, 4, 4, 1, bias = False)
        else:
            self.cate_classify = None

    def forward(self, image):
        img_embedding = self.encode_img(image)
        #(60,3,64,64) -> (60,992,4,4)
        return img_embedding
    
class STAGE1_D_SEG(nn.Module):
    def __init__(self, use_categories = True):
        super(STAGE1_D_SEG, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.CONDITION_DIM
        self.text_dim = cfg.TEXT.DIMENSION
        self.label_num = cfg.LABEL_NUM
        self.define_module(use_categories)

    def define_module(self, use_categories):
        ndf, nef = self.df_dim, self.ef_dim
        self.encode_img = nn.Sequential(
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*2) x 16 x 16
            spectral_norm(nn.Conv2d(ndf*2, ndf * 4, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*4) x 8 x 8
            spectral_norm(nn.Conv2d(ndf*4, ndf * 8, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 8),
            # state size (ndf * 8) x 4 x 4)
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.seq_consisten_model = None
        self.get_cond_logits = D_GET_LOGITS(int(ndf), nef + self.text_dim + self.label_num)
        self.get_uncond_logits = None

        if use_categories:
            self.cate_classify = nn.Conv2d(ndf * 8, self.label_num, 4, 4, 1, bias = False)
        else:
            self.cate_classify = None

    def forward(self, image):
        img_embedding = self.encode_img(image)
        #(60,3,64,64) -> (60,992,4,4)
        return img_embedding
    
class STAGE1_D_STY_V2(nn.Module):
    def __init__(self):
        super(STAGE1_D_STY_V2, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.CONDITION_DIM
        self.text_dim = cfg.TEXT.DIMENSION
        self.label_num = cfg.LABEL_NUM
        self.define_module()

    def define_module(self):
        ndf, nef = self.df_dim, self.ef_dim
        self.encode_img = nn.Sequential(
            spectral_norm(nn.Conv2d(3, ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*2) x 16 x 16
            spectral_norm(nn.Conv2d(ndf*2, ndf * 4, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*4) x 8 x 8
            spectral_norm(nn.Conv2d(ndf*4, ndf * 8, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 8),
            # state size (ndf * 8) x 4 x 4)
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.seq_consisten_model = None
        if cfg.USE_SEQ_CONSISTENCY:
            self.seq_consisten_model = VideoEncoder()
            # checkpoint = torch.load('logs/consistencybaseline_0.5/model.pt')
            # self.seq_consisten_model.load_state_dict(checkpoint['model'].state_dict())


        self.get_cond_logits = D_GET_LOGITS(ndf, nef + self.text_dim + self.label_num)
        self.get_uncond_logits = None
        self.cate_classify = None

    def forward(self, story):
        N, C, video_len, W, H = story.shape
        story = story.permute(0,2,1,3,4)
        story = story.contiguous().view(-1, C,W,H)
        story_embedding = torch.squeeze(self.encode_img(story))
        _, C1, W1, H1 = story_embedding.shape
        story_embedding = story_embedding.view(N,video_len, C1, W1, H1)
        story_embedding = story_embedding.mean(1).squeeze()
        return story_embedding

"""
class GET_LOGITS():
    def __init__(self):
        super(GET_LOGITS, self).__init__()
        self.project_dim = cfg.GAN.TEXT_CYC_DIS_PROJECT_DIM # 100
        self.define_module()
        
    def define_module(self):
        self.get_logits = nn.Sequential(
            nn.Linear(self.project_dim,1),
            nn.Sigmoid())
        
    def forward(self, input):
        return self.get_logits(input).view(-1)
        
    
class STAGE1_D_TextCyc(nn.Module):
    def __init__(self):
        super(STAGE1_D_TextCyc, self).__init__()
        self.text_dim = cfg.TEXT.DIMENSION # 356
        self.project_dim = cfg.GAN.TEXT_CYC_DIS_PROJECT_DIM # 100
        self.define_module()
        
    def define_module(self):
        self.embedding = nn.Sequential(
            nn.Linear(self.text_dim, self.project_dim),
            nn.BatchNorm1d(self.project_dim),
            nn.LeakyReLU(0.2, inplace=True))
        
        #self.get_uncond_logits = nn.Sequential(
        #    nn.Linear(self.project_dim,1),
        #    nn.Sigmoid())
        self.get_uncond_logits = GET_LOGITS()
    def forward(self, text):
        return self.embedding(text)"""
        
    

if __name__ == "__main__":
    # img = torch.randn(3, 3, 5, 64, 64).cuda()
    motion_input, content_input = torch.randn(5, 366).cuda(), torch.randn(5, 5, 356).cuda()
    m = StoryGAN(5).cuda() #downBlock(3, 128)
    o = m.sample_images(motion_input, content_input)
    print(o[1].shape)