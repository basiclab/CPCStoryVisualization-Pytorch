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
#from frechet_video_distance import frechet_video_distance as fvd
from .frechet_video_distance import calculate_fvd, create_id3_embedding, preprocess

import torch.utils.data
from torchvision.datasets import ImageFolder
from torchvision import transforms
import functools
import PIL
import re
import pdb
import argparse
from tqdm import tqdm
from .loader import VideoGenerateDataset
import random
import os
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '0'

# Number of videos must be divisible by 16.
#NUMBER_OF_VIDEOS = 320
#VIDEO_LENGTH = 9

def calculate_fvd_from_inference_result(gen_path, ref_path='./Evaluation/ref', num_of_video=16, video_length=10):

  VIDEO_LENGTH = video_length
  print('{}'.format(video_length))
  base_ref = VideoGenerateDataset(ref_path, min_len=VIDEO_LENGTH)
  base_tar = VideoGenerateDataset(gen_path, min_len=VIDEO_LENGTH)

  bs = num_of_video
  assert bs%16 == 0

  videoloader_ref = torch.utils.data.DataLoader(
    base_ref, batch_size=bs,  #len(videodataset),
    drop_last=True, shuffle=False)
  videoloader_tar = torch.utils.data.DataLoader(
    base_tar, batch_size=bs,  #len(videodataset),
    drop_last=True, shuffle=False)

  with tqdm(total=len(videoloader_ref), dynamic_ncols=True) as pbar:
    for i, data in enumerate(videoloader_ref):
      images_ref = data.numpy()
      break
    for i, data in enumerate(videoloader_tar):
      images_tar = data.numpy()
      break

  with tf.Graph().as_default():
    ref_tf = tf.convert_to_tensor(images_ref, dtype=tf.uint8)
    tar_tf = tf.convert_to_tensor(images_tar, dtype=tf.uint8)

    first_set_of_videos = ref_tf #14592
    second_set_of_videos = tar_tf

    result = calculate_fvd(
        create_id3_embedding(preprocess(first_set_of_videos,
                                                (224, 224)), bs),
        create_id3_embedding(preprocess(second_set_of_videos,
                                                (224, 224)), bs))

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.tables_initializer())
      return sess.run(result)
        


