import torch
from collections import defaultdict
import contextlib
import copy
import numpy as np
import importlib.util
import math
import os
import sys
from typing import Callable, List
import torchvision.utils as vutils
import warnings
import torch.nn.functional as F
import math
import random
from torch import autograd
from PIL import Image, ImageSequence
from collections import Counter
from torchtext.vocab import Vocab
import errno
import PIL
from copy import deepcopy
import pdb
from torch.nn import init
import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.autograd import Variable
from sklearn.metrics import accuracy_score


def check_is_order(sequence):
	return (np.diff(sequence)>=0).all()

def read_gif_file(filename, seek_pos=1):
	img = Image.open(filename)
	try:
		img.seek(seek_pos)
	except EOFError:
		return img, 1
	return img, seek_pos

def gelu_accurate(x):
	if not hasattr(gelu_accurate, "_a"):
		gelu_accurate._a = math.sqrt(2 / math.pi)
	return 0.5 * x * (1 + torch.tanh(gelu_accurate._a * (x + 0.044715 * torch.pow(x, 3))))

def gelu(x: torch.Tensor) -> torch.Tensor:
	if hasattr(torch.nn.functional, 'gelu'):
		return torch.nn.functional.gelu(x.float()).type_as(x)
	else:
		return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def clip_grad_norm_(tensor, max_norm):
	grad_norm = item(torch.norm(tensor))
	if grad_norm > max_norm > 0:
		clip_coef = max_norm / (grad_norm + 1e-6)
		tensor.mul_(clip_coef)
	return grad_norm


def sample_gumbel(shape, eps=1e-20):
	u = torch.rand(shape)
	if torch.cuda.is_available():
		u = u.cuda()
	return -torch.log(-torch.log(u + eps) + eps)


def gumbel_softmax(logits, temperature, st_mode=False):
	"""
	Gumble Softmax

	Args:
		logits: float tensor, shape = [*, n_class]
		temperature: float
		st_mode: boolean, Straight Through mode
	Returns:
		return: gumbel softmax, shape = [*, n_class]
	"""
	logits = logits + sample_gumbel(logits.size())
	return softmax(logits, temperature, st_mode)


def softmax(logits, temperature=1, st_mode=False):
	"""
	Softmax

	Args:
		logits: float tensor, shape = [*, n_class]
		st_mode: boolean, Straight Through mode
	Returns:
		return: gumbel softmax, shape = [*, n_class]
	"""
	y = torch.nn.functional.softmax(logits, dim=-1)
	if st_mode:
		return straight_through_estimate(y)
	else:
		return y

def straight_through_estimate(p):
	shape = p.size()
	ind = p.argmax(dim=-1)
	p_hard = torch.zeros_like(p).view(-1, shape[-1])
	p_hard.scatter_(1, ind.view(-1, 1), 1)
	p_hard = p_hard.view(*shape)
	return ((p_hard - p).detach() + p)

def images_to_numpy(tensor):
	generated = tensor.data.cpu().numpy().transpose(1,2,0)
	generated[generated < -1] = -1
	generated[generated > 1] = 1
	generated = (generated + 1) / 2 * 255
	return generated.astype('uint8')

def create_random_shuffle(stories,  random_rate=0.5, sample_video_len=3):
	o3n_data, label = [], []
	stories = stories.cpu()
	for result in stories:
		video_len = result.shape[1]
		print(video_len, result.shape)
		is_fake = 1 if random_rate > np.random.random() else 0
		if is_fake == 0:
			o3n_data.append(result.clone())
		else:
			random_sequence = random.sample(range(video_len), video_len)
			while (check_is_order(random_sequence)): # make sure not sorted
				np.random.shuffle(random_sequence)
			o3n_data.append(result[:, list(random_sequence), :, :].clone())
		label.append(is_fake)

	return torch.stack(o3n_data, 0), torch.from_numpy(np.array(label)).float()


def save_story_results(ground_truth, images, label ,epoch, image_dir, video_len = 5, test = False,
	writer=None, steps=0):
	all_images = []
	for i in range(images.shape[0]):
		all_images.append(vutils.make_grid(images[i], nrow=video_len))

	all_images= vutils.make_grid(all_images, 1)
	all_images = images_to_numpy(all_images)

	if ground_truth is not None:
		gts = []
		for i in range(ground_truth.shape[0]):
			gts.append(vutils.make_grid(ground_truth[i], nrow=video_len))
		gts = vutils.make_grid(gts, 1)
		gts = images_to_numpy(gts)
		all_images = np.concatenate([all_images, gts], axis = 1)
	if writer and (steps+1) % 100 == 0:
		writer.add_image('Image', np.transpose(all_images, (2, 0, 1))/255, steps+1)
	output = Image.fromarray(all_images)
	if not test:
		output.save('%s/fake_samples_epoch_%03d.png' % (image_dir, epoch) )
	else:
		output.save('%s/test_samples_%03d.png' % (image_dir, epoch) )
	return

def calc_gradient_penalty(net_D, real, fake, cond, gpus, gp_center=1):
	 
	alpha = torch.rand(real.size(0), *([1]*(len(real.shape)-1)) ).cuda()
	alpha = alpha.expand(real.size())

	interpolates = alpha * real.detach() + (1 - alpha) * fake.detach()
	interpolates.requires_grad = True
	disc_interpolates_features = nn.parallel.data_parallel(net_D, (interpolates), gpus)

	inputs = (disc_interpolates_features, cond)
	disc_interpolates = nn.parallel.data_parallel(net_D.get_cond_logits, inputs, gpus)

	ones = torch.ones(disc_interpolates.size()).cuda()
	gradients = autograd.grad(
		outputs=disc_interpolates, inputs=interpolates,
		grad_outputs=ones,
		create_graph=True, retain_graph=True, only_inputs=True)[0]
	gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=[1, 2]))
	gradient_penalty = ((gradients_norm - gp_center) ** 2).mean()
	return gradient_penalty

def KL_loss(mu, logvar):
	# -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
	KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
	KLD = torch.mean(KLD_element).mul_(-0.5)
	return KLD


def compute_generator_loss(netD, fake_imgs, real_labels, fake_catelabels, conditions, gpus):
	ratio = 0.4
	criterion = nn.BCELoss()
	cate_criterion =nn.MultiLabelSoftMarginLoss()
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
		errD_fake = errD_fake + ratio * cate_criterion(cate_logits, fake_catelabels)
		acc = accuracy_score(fake_catelabels.cpu().data.numpy().astype('int32'),
			(cate_logits.cpu().data.numpy() > 0.5).astype('int32'))
	return errD_fake, acc


#############################
def weights_init(m):
	classname = m.__class__.__name__
	# if classname.find('Conv') != -1:
	# 	m.weight.data.normal_(0.0, 0.02)
	if classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)
	elif classname.find('Linear') != -1:
		m.weight.data.normal_(0.0, 0.02)
		if m.bias is not None:
			m.bias.data.fill_(0.0)


#############################
def save_img_results(data_img, fake,epoch, image_dir, writer=None, steps=0, num=10):
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
		if writer:
			x = vutils.make_grid(fake.data, normalize=True)
			writer.add_image('Image', x, step+1)
		vutils.save_image(
			fake.data, '%s/lr_fake_samples_epoch_%03d.png' %
			(image_dir, epoch), normalize=True)

##########################\
def images_to_numpy(tensor):
	generated = tensor.data.cpu().numpy().transpose(1,2,0)
	generated[generated < -1] = -1
	generated[generated > 1] = 1
	generated = (generated + 1) / 2 * 255
	return generated.astype('uint8')

def save_story_results(ground_truth, images, epoch, image_dir, video_len = 5, test = False, writer=None, steps=0):
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
	if writer and (steps+1) % 100 == 0:
		writer.add_image('Image', np.transpose(all_images, (2, 0, 1))/255, steps+1)
	output = PIL.Image.fromarray(all_images)
	if not test:
		output.save('%s/fake_samples_epoch_%03d.png' % (image_dir, epoch) )
	else:
		output.save('%s/test_samples_%03d.png' % (image_dir, epoch) )
	return

def get_multi_acc(predict, real):
	predict = 1/(1+np.exp(-predict))
	correct = 0
	for i in range(predict.shape[0]):
		for j in range(predict.shape[1]):
			if real[i][j] == 1 and predict[i][j]>=0.5 :
				correct += 1
	acc = correct / float(np.sum(real))
	return acc


def save_model(netG, netD_im, netD_st, epoch, model_dir):
	torch.save(
		netG.state_dict(),
		'%s/netG_epoch_%d.pth' % (model_dir, epoch))
	torch.save(
		netD_im.state_dict(),
		'%s/netD_im_epoch_last.pth' % (model_dir))
	torch.save(
		netD_st.state_dict(),
		'%s/netD_st_epoch_last.pth' % (model_dir))
	print('Save G/D models')


def mkdir_p(path):
	try:
		os.makedirs(path)
	except OSError as exc:  # Python >2.5
		if exc.errno == errno.EEXIST and os.path.isdir(path):
			pass
		else:
			raise

def save_test_samples(netG, dataloader, save_path, writer=None, steps=0):
	print('Generating Test Samples...')
	labels = []
	gen_images = []
	real_images = []
	netG.eval()
	with torch.no_grad():
		for i, batch in enumerate(dataloader, 0):
			real_cpu = batch['images']
			motion_input = batch['description']
			content_input = batch['description']
			catelabel = batch['labels']
			real_imgs = Variable(real_cpu)
			motion_input = Variable(motion_input)
			content_input = Variable(content_input)
			if next(netG.parameters()).is_cuda:
				real_imgs = real_imgs.cuda()
				motion_input = motion_input.cuda()
				content_input = content_input.cuda()
				catelabel = catelabel.cuda()
                motion_input = torch.cat((motion_input, catelabel), 2)  

			_, fake, _,_,_,_,_ = netG.sample_videos(motion_input, content_input)
			save_story_results(real_cpu, fake, i, save_path, writer=writer, steps=steps)
			break

	for i, batch in enumerate(dataloader, 0):
		if i>10:
			break
		real_cpu = batch['images']
		motion_input = batch['description']
		content_input = batch['description']
		catelabel = batch['labels']
		real_imgs = Variable(real_cpu)
		motion_input = Variable(motion_input)
		content_input = Variable(content_input)
		if next(netG.parameters()).is_cuda:
			real_imgs = real_imgs.cuda()
			motion_input = motion_input.cuda()
			content_input = content_input.cuda()
			catelabel = catelabel.cuda()
            motion_input = torch.cat((motion_input, catelabel), 2) # (12,5,365)
		_, fake, _,_,_,_,_ = netG.sample_videos(motion_input, content_input)
		save_story_results(real_cpu, fake, i, save_path, 5, True, writer=writer, steps=steps)
		break
