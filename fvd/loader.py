# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example code that computes FVD for some empty frames.
The FVD for this setup should be around 131.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf

import torch.utils.data
from torchvision.datasets import ImageFolder
from torchvision import transforms
import functools
import PIL
import re
import pdb
from tqdm import tqdm
import random
import os
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '0'

# Number of videos must be divisible by 16.
#NUMBER_OF_VIDEOS = 320
#VIDEO_LENGTH = 9

class VideoFolderDataset(torch.utils.data.Dataset):
  def __init__(self, folder, counter=None, cache_file=None, degree=[0], min_len=4, target=False):
    self.folder = folder
    self.storys = []
    dataset = ImageFolder(folder)

    path_story_cache = os.path.join(folder, cache_file)
    if cache_file is not None and os.path.exists(path_story_cache):
      print('Load cache file: {}'.format(path_story_cache))
      self.storys = np.load(path_story_cache, allow_pickle=True, encoding='latin1')
    else:
      possible_order = []
      for deg in degree:
        possible_order += get_degree_dict(length=int(min_len))[deg]
      order = random.choice(possible_order)
        
      for idx, (im, _) in enumerate(tqdm(dataset, desc='Counting total number of frame')):
        img_path, _ = dataset.imgs[idx]
        episo_name = '_'.join(img_path.split('/')[-1].split('_')[:-1])
        id = int(re.split('/|_|\.', img_path)[-2])
        num_frames = counter[episo_name]
        if id > num_frames-min_len:
          continue
        story = [img_path.split('/')[-1]]
        for i in range(min_len-1):
          id += 1
          story += ['{}_{}.png'.format(episo_name, id)]
        if degree:
          ordered_story = [story[j] for j in order]
          self.storys.append(ordered_story)
        else:
          self.storys.append(story)

      np.save('{}/{}'.format(folder, cache_file), self.storys)
      print('Generate and Save the cache file: {}'.format(cache_file))
    self.target = target
    print('Total number of clips: {}'.format(len(self.storys)))
    print('Target: {}'.format(target))

    """
    path_img_cache = "{}/img_cache{}.npy".format(cache_file, min_len)
    path_follow_cache = "{}/following_cache{}.npy".format(cache_file, min_len)
    self.images = np.load(path_img_cache,allow_pickle=True, encoding = 'latin1')
    self.followings = np.load(path_follow_cache, allow_pickle=True, encoding = 'latin1')"""
  def sample_image(self, im):
    shorter, longer = min(im.size[0], im.size[1]), max(im.size[0], im.size[1])
    video_len = longer // shorter
    se = np.random.randint(0,video_len, 1)[0]
    return im.crop((0, se * shorter, shorter, (se+1)*shorter))

  def __getitem__(self, item):
    # return a training list
    img_list = self.storys[item]
    images = []
    if self.target:
      for i in range(len(img_list)):
        img = img_list[i]
        img = img.split('\\')[-1]
        im = PIL.Image.open(os.path.join(self.folder, 'images', img))
        im = im.convert('RGB')
        images.append(np.expand_dims(np.array(im), axis=0))
    else:
      for img in img_list:
        img = img.split('\\')[-1]
        im = PIL.Image.open(os.path.join(self.folder, 'images', img))
        im = im.convert('RGB')
        images.append(np.expand_dims(np.array(im), axis=0))
    images = np.concatenate(images, axis=0)
    return images

    """
    lists = [self.images[item]]
    for i in range(len(self.followings[item])):
      lists.append(str(self.followings[item][i]))
    return lists
    """

  def __len__(self):
    #return len(self.images)
    return len(self.storys)


class VideoGenerateDataset(torch.utils.data.Dataset):
  def __init__(self, folder, min_len):
    self.folder = folder
    self.storys = []
    story = []

    tot_imgs = len(os.listdir(folder))
    for i in range(tot_imgs):
      i += 1
      story += [i]
      if i % min_len == 0:
        self.storys += [story]
        story = []
    print('Total number of clips: {}'.format(len(self.storys)))

  def sample_image(self, im):
    shorter, longer = min(im.size[0], im.size[1]), max(im.size[0], im.size[1])
    video_len = longer // shorter
    se = np.random.randint(0,video_len, 1)[0]
    return im.crop((0, se * shorter, shorter, (se+1)*shorter))

  def __getitem__(self, item):
    # return a training list
    img_list = self.storys[item]
    images = []
    for img in img_list:
      img = '{}.png'.format(img)
      im = PIL.Image.open(os.path.join(self.folder, img))
      im = im.convert('RGB')
      images.append(np.expand_dims(np.array(im), axis=0))
    images = np.concatenate(images, axis=0)
    return images

  def __len__(self):
    #return len(self.images)
    return len(self.storys)


class VideoDataset(torch.utils.data.Dataset):
  def __init__(self, dataset, transform=None):
    self.dir_path = dataset.dir_path
    self.dataset = dataset
    self.transforms = transform

  def __getitem__(self, item):
    lists = self.dataset[item]
    image = []
    subs = []
    des = []
    text = []
    for v in lists:
      id = v.replace('.png','')
      path = self.dir_path + id + '.png'
      im = PIL.Image.open(path)
      im = im.convert('RGB')
      image.append( np.expand_dims(np.array(self.dataset.sample_image(im)), axis = 0) )
      image = np.concatenate(image, axis = 0)
      #image = self.transforms(image)
      return {'images': image}

  def __len__(self):
    return len(self.dataset.images)
