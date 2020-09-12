#!/usr/bin/env python3

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy import linalg
import pytorch_ssim


def ssim_score(imgs):

	dataloader = DataLoader(imgs, batch_size=1, shuffle=False)
	avg_ssim = 0
	ind = 0
	for images in dataloader:

		r_imgs, g_imgs = images
		#print(r_imgs.size())
		r_imgs = r_imgs.type(torch.FloatTensor).to(0)
		g_imgs = g_imgs.type(torch.FloatTensor).to(0)
		for r_img,g_img in zip(r_imgs, g_imgs):
			ind += 1
			ssim_value = pytorch_ssim.ssim(r_img, g_img)
			avg_ssim += ssim_value.item()
	return avg_ssim/ind
