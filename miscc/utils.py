import os
import errno
import numpy as np
import PIL
from copy import deepcopy
from miscc.config import cfg
import pdb
from torch.nn import init
import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.autograd import Variable
import random
from tqdm import tqdm

#############################
def check_is_order(sequence):
    return (np.diff(sequence)>=0).all()

def create_random_shuffle(stories,  random_rate=0.5):
    o3n_data, labels = [], []
    device = stories.device
    stories = stories.cpu()
    story_size = len(stories)
    for idx, result in enumerate(stories):
        video_len = result.shape[1]
        label = 1 if random_rate > np.random.random() else 0
        if label == 0:
            o3n_data.append(result.clone())
        else:
            random_sequence = random.sample(range(video_len), video_len)
            while (check_is_order(random_sequence)): # make sure not sorted
                np.random.shuffle(random_sequence)
            shuffled_story = result[:, list(random_sequence), :, :].clone()
            story_size_idx = random.randint(0, story_size-1)
            if story_size_idx != idx:
                story_mix = random.sample(range(video_len), 1)
                shuffled_story[:, story_mix, :, : ] = stories[story_size_idx, :, story_mix, :, :].clone()
            o3n_data.append(shuffled_story)
        labels.append(label)

    order_labels = Variable(torch.from_numpy(np.array(labels)).float(), requires_grad=True).detach()
    shuffle_imgs = Variable(torch.stack(o3n_data, 0), requires_grad=True)
    return shuffle_imgs.to(device), order_labels.to(device)

##### Discriminartor loss

def compute_discriminator_loss(netD, real_imgs, fake_imgs,
                               real_labels, fake_labels,real_catelabels,
                               conditions, gpus):
    criterion = nn.BCELoss()
    cate_criterion =nn.MultiLabelSoftMarginLoss()
    batch_size = real_imgs.size(0)
    fake = fake_imgs.detach()

    if conditions is None:
        # For Text Cycle Discriminator
        real_features = nn.parallel.data_parallel(netD, (real_imgs), gpus)
        fake_features = nn.parallel.data_parallel(netD, (fake), gpus)
        real_logits = nn.parallel.data_parallel(netD.get_uncond_logits, (real_features), gpus)
        fake_logits = nn.parallel.data_parallel(netD.get_uncond_logits, (fake_features), gpus)

        errD_real = criterion(real_logits, real_labels)
        errD_fake = criterion(fake_logits, fake_labels)
        errD = errD_real + errD_fake
        return errD, errD_real.data, errD_fake.data
    
    else:
        cond = conditions.detach() # the m_mu (60,489)
        real_features = nn.parallel.data_parallel(netD, (real_imgs), gpus)
        fake_features = nn.parallel.data_parallel(netD, (fake), gpus)
        
        # real pairs
        inputs = (real_features, cond)
        real_logits = nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
        errD_real = criterion(real_logits, real_labels)
        # wrong pairs
        inputs = (real_features[:(batch_size-1)], cond[1:])
        wrong_logits = nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
        errD_wrong = criterion(wrong_logits, fake_labels[1:])
        # fake pairs
        inputs = (fake_features, cond)
        fake_logits = nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
        errD_fake = criterion(fake_logits, fake_labels)

    if netD.get_uncond_logits is not None:
        real_logits = \
            nn.parallel.data_parallel(netD.get_uncond_logits,
                                      (real_features), gpus)
        fake_logits = \
            nn.parallel.data_parallel(netD.get_uncond_logits,
                                      (fake_features), gpus)
        uncond_errD_real = criterion(real_logits, real_labels)
        uncond_errD_fake = criterion(fake_logits, fake_labels)
        #
        errD = ((errD_real + uncond_errD_real) / 2. +
                (errD_fake + errD_wrong + uncond_errD_fake) / 3.)
        errD_real = (errD_real + uncond_errD_real) / 2.
        errD_fake = (errD_fake + uncond_errD_fake) / 2.
    else:
        errD = errD_real + (errD_fake + errD_wrong) * 0.5

    acc = 0
    if netD.cate_classify is not None:
        cate_logits = nn.parallel.data_parallel(netD.cate_classify, real_features, gpus)
        cate_logits = cate_logits.squeeze()
        errD = errD + 1.0 * cate_criterion(cate_logits, real_catelabels)
        acc = get_multi_acc(cate_logits.cpu().data.numpy(), real_catelabels.cpu().data.numpy())

    consistency_loss_val = 0
    if netD.seq_consisten_model:
        bce_loss = nn.BCEWithLogitsLoss()
        B = fake_imgs.shape[0]
        shuffle_story_input, order_labels = create_random_shuffle(real_imgs)
        # fake_labels = torch.zeros(B)
        order_logits = nn.parallel.data_parallel(netD.seq_consisten_model, shuffle_story_input, gpus)
        order_labels = order_labels.cuda()
        real_consistent_loss = bce_loss(order_logits, order_labels.unsqueeze(-1) )
        # fake_consistent_loss = criterion(fake_labels, fake_imgs)
        consistency_loss = real_consistent_loss
        errD += cfg.CONSISTENCY_RATIO * consistency_loss
        consistency_loss_val = consistency_loss.item()
    return errD, errD_real.data, errD_wrong.data, errD_fake.data, acc, consistency_loss_val

##### Generating Loss #####
def compute_generator_loss(netD, fake_imgs, real_imgs, real_labels, fake_catelabels, conditions, gpus):
    criterion = nn.BCELoss()
    cate_criterion =nn.MultiLabelSoftMarginLoss()
    
    if conditions is None:
        fake_features = nn.parallel.data_parallel(netD, (fake_imgs), gpus)
        fake_logits = nn.parallel.data_parallel(netD.get_uncond_logits, (fake_features), gpus)
        errD_fake = criterion(fake_logits, real_labels)
        return errD_fake
    else:
        cond = conditions.detach()
        fake_features = nn.parallel.data_parallel(netD, (fake_imgs), gpus)
        # fake pairs
        inputs = (fake_features, cond)
        fake_logits = nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
        errD_fake = criterion(fake_logits, real_labels)
    if netD.get_uncond_logits is not None:
        fake_logits = \
            nn.parallel.data_parallel(netD.get_uncond_logits,
                                      (fake_features), gpus)
        uncond_errD_fake = criterion(fake_logits, real_labels)
        errD_fake += uncond_errD_fake
    acc = 0
    if netD.cate_classify is not None:
        cate_logits = nn.parallel.data_parallel(netD.cate_classify, fake_features, gpus)
        cate_logits = cate_logits.squeeze()
        errD_fake = errD_fake + 1.0 * cate_criterion(cate_logits, fake_catelabels)
        acc = get_multi_acc(cate_logits.cpu().data.numpy(), fake_catelabels.cpu().data.numpy())
    consistency_loss_val = 0
    if netD.seq_consisten_model:
        mse_loss = nn.MSELoss()
        # bce_loss = nn.BCEWithLogitsLoss()
        # B = fake_imgs.shape[0]
        # fake_labels = torch.zeros(B).cuda()
        # order_logits = nn.parallel.data_parallel(netD.seq_consisten_model, fake_imgs, gpus)

        # fake_consistent_loss = bce_loss(order_logits, fake_labels.unsqueeze(-1) )
        # consistency_loss = fake_consistent_loss
        # errD_fake += 1.0 * consistency_loss
        real_logits = nn.parallel.data_parallel(netD.seq_consisten_model, real_imgs, gpus)
        fake_logits = nn.parallel.data_parallel(netD.seq_consisten_model, fake_imgs, gpus)
        consistency_loss = mse_loss(fake_logits, real_logits.detach())
        errD_fake += cfg.CONSISTENCY_RATIO * consistency_loss
        consistency_loss_val = consistency_loss.item()

    return errD_fake, acc, consistency_loss_val


def compute_cyc_loss_img(loss_fn, st_cyc_imgs, st_real_imgs):
    loss = loss_fn(st_cyc_imgs, st_real_imgs) # mse
    #loss = F.binary_cross_entropy(st_cyc_imgs.view(-1,64*64), st_real_imgs.view(-1,64*64)) # x_ent
    return loss


def compute_cyc_loss_txt(loss_fn, st_motion_cyc, st_motion_input):
    loss = loss_fn(st_motion_cyc, st_motion_input).mean()
    return loss

def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD

#############################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


#############################
def save_img_results(data_img, fake, texts, epoch, image_dir):
    num = cfg.VIS_COUNT
    fake = fake[0:num]
    # data_img is changed to [0,1]
    if data_img is not None:
        data_img = data_img[0:num]
        vutils.save_image(
            data_img, '%s/real_samples_epoch_%03d.png' % 
            (image_dir, epoch), normalize=True)
        # fake.data is still [-1, 1]
        vutils.save_image(
            fake.data, '%s/fake_samples_epoch_%03d.png' %
            (image_dir, epoch), normalize=True)
    else:
        vutils.save_image(
            fake.data, '%s/lr_fake_samples_epoch_%03d.png' %
            (image_dir, epoch), normalize=True)

    if texts is not None:
        fid = open('%s/lr_fake_samples_epoch_%03d.txt' % (image_dir, epoch), 'wb')
        for i in range(num):
            fid.write(str(i) + ':' + texts[i] + '\n')
        fid.close()

##########################\
def images_to_numpy(tensor):
    generated = tensor.data.cpu().numpy().transpose(1,2,0)
    generated[generated < -1] = -1
    generated[generated > 1] = 1
    generated = (generated + 1) / 2 * 255
    return generated.astype('uint8')

def save_story_results(ground_truth, images, texts, name, image_dir, step=0, lr = False):
    video_len = cfg.VIDEO_LEN
    all_images = []
    for i in range(images.shape[0]):
        all_images.append(vutils.make_grid(torch.transpose(images[i], 0,1), video_len))
    all_images= vutils.make_grid(all_images, 1)
    all_images = images_to_numpy(all_images)
    
    if ground_truth is not None:
        gts = []
        for i in range(ground_truth.shape[0]):
            gts.append(vutils.make_grid(torch.transpose(ground_truth[i], 0,1), video_len))
        gts = vutils.make_grid(gts, 1)
        gts = images_to_numpy(gts)
        all_images = np.concatenate([all_images, gts], axis = 1)

    """
    if seg_images is not None:
        seg_images.view()
        video_len = cfg.VIDEO_LEN
        st_bs = cfg.TRAIN.ST_BATCH_SIZE
        size = cfg.SESIZE
        seg_images = seg_images.reshape(st_bs, video_len, -1, size, size)
        gts = []
        for i in range(seg_images.shape[0]):
            gts.append(vutils.make_grid(torch.transpose(seg_images[i], 0,1), video_len))
        gts = vutils.make_grid(gts, 1)
        gts = images_to_numpy(gts)"""

    output = PIL.Image.fromarray(all_images)
    #if lr:
    #    output.save('{}/lr_samples_{}_{}.png'.format(image_dir, name, step))
    #else:
    #    output.save('{}/fake_samples_{}_{}.png'.format(image_dir, name, step))

    if texts is not None:
        fid = open('{}/fake_samples_{}.txt'.format(image_dir, name), 'w')
        for idx in range(images.shape[0]):
            fid.write(str(idx) + '--------------------------------------------------------\n')
            for i in range(len(texts)):
                fid.write(texts[i][idx] +'\n' )
            fid.write('\n\n')
        fid.close()
    return all_images

def save_image_results(ground_truth, images, size=cfg.IMSIZE):
    video_len = cfg.VIDEO_LEN
    st_bs = cfg.TRAIN.ST_BATCH_SIZE
    images = images.reshape(st_bs, video_len, -1, size, size)
    all_images = []
    for i in range(images.shape[0]):
        all_images.append(vutils.make_grid(images[i], video_len))
    all_images= vutils.make_grid(all_images, 1)
    all_images = images_to_numpy(all_images)
    
    if ground_truth is not None:
        ground_truth = ground_truth.reshape(st_bs, video_len, -1, size, size)
        gts = []
        for i in range(ground_truth.shape[0]):
            gts.append(vutils.make_grid(ground_truth[i], video_len))
        gts = vutils.make_grid(gts, 1)
        gts = images_to_numpy(gts)
        all_images = np.concatenate([all_images, gts], axis = 1)
    #output = PIL.Image.fromarray(all_images)
    return all_images

def save_all_img(images, count, image_dir):
    bs, size_c, v_len, size_w, size_h = images.shape
    for b in range(bs):
        imgs = images[b].transpose(0,1)
        for i in range(v_len):
            count += 1
            png_name = os.path.join(image_dir, "{}.png".format(count))
            vutils.save_image(imgs[i], png_name)
    return count

def get_multi_acc(predict, real):
    predict = 1/(1+np.exp(-predict))
    correct = 0
    for i in range(predict.shape[0]):
        for j in range(predict.shape[1]):
            if real[i][j] == 1 and predict[i][j]>=0.5 :
                correct += 1
    acc = correct / float(np.sum(real))
    return acc

def save_model(netG, netD_im, netD_st, netD_se, epoch, model_dir, whole=False):
    if whole == True:
        # Save the whole model
        torch.save(netG, '%s/netG.pkl' % (model_dir))
        torch.save(netD_im, '%s/netD_im.pkl' % (model_dir))
        torch.save(netD_st, '%s/netD_st.pkl' % (model_dir))
        if netD_se is not None:
            torch.save(netD_se, '%s/netD_se.pkl' % (model_dir))
        print('Save G/D model')
        return
    torch.save(netG.state_dict(),'%s/netG_epoch_%d.pth' % (model_dir, epoch))
    torch.save(netD_im.state_dict(),'%s/netD_im_epoch_last.pth' % (model_dir))
    torch.save(netD_st.state_dict(),'%s/netD_st_epoch_last.pth' % (model_dir))
    if netD_se is not None:
        torch.save(netD_se.state_dict(),'%s/netD_se_epoch_last.pth' % (model_dir))
    print('Save G/D models')

def mkdir_p(path):
    os.makedirs(path, exist_ok=True)

def save_test_samples(netG, dataloader, save_path):
    print('Generating Test Samples...')
    save_images = []
    save_labels = []
    for i, batch in enumerate(dataloader, 0):
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
        _, fake, _,_,_,_,_ = netG.sample_videos(motion_input, content_input)
        save_story_results(real_cpu, fake, batch['text'], '{:03d}'.format(i), save_path)
        save_images.append(fake.cpu().data.numpy())
        save_labels.append(catelabel.cpu().data.numpy())
    save_images = np.concatenate(save_images, 0)
    save_labels = np.concatenate(save_labels, 0)
    np.save(save_path + '/images.npy', save_images)
    np.save(save_path + '/labels.npy', save_labels)

def save_train_samples(netG, dataloader, save_path):
    print('Generating Train Samples...')
    save_images = []
    save_labels = []
    for i, batch in enumerate(dataloader, 0):
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
        _, fake, _,_,_,_,_ = netG.sample_videos(motion_input, content_input)
        save_story_results(real_cpu, fake, batch['text'], '{:05d}'.format(i), save_path)
        save_images.append(fake.cpu().data.numpy())
        save_labels.append(catelabel.cpu().data.numpy())
    save_images = np.concatenate(save_images, 0)
    save_labels = np.concatenate(save_labels, 0)
    np.save(save_path + '/images.npy', save_images)
    np.save(save_path + '/labels.npy', save_labels)


def inference_samples(netG, dataloader, save_path):
    print('Generate and save images...')
    #save_images = []
    #save_labels = []
    mkdir_p(save_path)
    mkdir_p('./Evaluation/ref')
    cnt_gen = 0
    cnt_ref = 0
    for i, batch in enumerate(tqdm(dataloader, desc='Saving')):
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
        _, fake, _,_,_,_,_ = netG.sample_videos(motion_input, content_input)
        cnt_gen = save_all_img(fake, cnt_gen, save_path)
        cnt_ref = save_all_img(real_imgs, cnt_ref, './Evaluation/ref')

  
def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count 


if __name__ == "__main__":
    test = torch.randn((14, 3, 5, 64,64))
    output, labels = create_random_shuffle(test)
    print(output.shape)
